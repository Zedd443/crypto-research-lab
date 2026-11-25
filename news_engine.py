"""Point-in-time news collector. No retrospective verification or fabricated arrival times."""
import argparse
import hashlib
import json
import sqlite3
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
import xml.etree.ElementTree as ET
import requests
import polars as pl

FEEDS = {
    "fed": "https://www.federalreserve.gov/feeds/press_all.xml",
    "bls": "https://www.bls.gov/feed/bls_latest.rss",
}
ALIASES = {
    "XAUUSDT": ["gold", "bullion"], "QQQUSDT": ["nasdaq", "qqq"],
    "SPYUSDT": ["s&p 500", "sp500", "spy"], "NVDAUSDT": ["nvidia", "nvda"],
    "AAPLUSDT": ["apple inc", "apple earnings", "aapl"],
}
def now_ms():
    return int(time.time()*1000)

def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()

def connect(root):
    root.mkdir(parents=True, exist_ok=True)
    db=sqlite3.connect(root/"news.sqlite", timeout=30)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("""CREATE TABLE IF NOT EXISTS news (
        id TEXT PRIMARY KEY, source TEXT NOT NULL, url TEXT NOT NULL,
        headline TEXT NOT NULL, published_ms INTEGER, received_ms INTEGER NOT NULL,
        available_ms INTEGER NOT NULL, source_type TEXT NOT NULL,
        verification_status TEXT NOT NULL, content_hash TEXT NOT NULL
    )""")
    db.execute("""CREATE TABLE IF NOT EXISTS sentiment (
        news_id TEXT NOT NULL, model TEXT NOT NULL, revision TEXT NOT NULL,
        scored_ms INTEGER NOT NULL, positive REAL, negative REAL, neutral REAL,
        PRIMARY KEY(news_id,model,revision)
    )""")
    return db

def parse_date(s):
    if not s:return None
    try:
        dt=parsedate_to_datetime(s)
        if dt.tzinfo is None:return None
        return int(dt.timestamp()*1000)
    except (ValueError, TypeError, OverflowError):
        return None

def ingest(db,source,url,headline,published_ms,received_ms,source_type="unverified"):
    headline=headline.strip()
    if not headline or not url:return
    h=digest(headline)
    # Different headline versions are separate events; repeat polling is idempotent.
    key=digest(source+"\n"+url+"\n"+h)
    available=max(received_ms,published_ms or received_ms)
    db.execute("INSERT OR IGNORE INTO news VALUES (?,?,?,?,?,?,?,?,?,?)",
        (key,source,url,headline,published_ms,received_ms,available,source_type,
         "primary_source" if source_type=="official_feed" else "unverified",h))

def collect(db):
    failures=[]
    for source,url in FEEDS.items():
        try:
            r=requests.get(url,timeout=30,headers={"User-Agent":"CryptoResearchLab/1.0 news research"})
            r.raise_for_status()
            received=now_ms()
            items=ET.fromstring(r.content).findall(".//item")
            if not items:raise ValueError("No RSS items")
            for item in items:
                ingest(db,source,item.findtext("link") or "",item.findtext("title") or "",
                       parse_date(item.findtext("pubDate")),received,"official_feed")
            db.commit()
            print(source,len(items),"items",flush=True)
        except Exception as e:
            db.rollback()
            failures.append(source)
            print(source,"FAILED",repr(e),flush=True)
    return failures

def import_jsonl(db,path):
    # Historical imports without actual receipt timestamps become available NOW,
    # not at claimed publication time. Keep provenance explicit.
    with path.open(encoding="utf-8") as f:
        for line in f:
            row=json.loads(line)
            received=row.get("received_ms")
            if received is None:received=now_ms()
            ingest(db,row["source"],row["url"],row["headline"],
                   row.get("published_ms"),int(received),"imported")
    db.commit()

def fng(root):
    r=requests.get("https://api.alternative.me/fng/",params={"limit":0},timeout=60)
    r.raise_for_status()
    rows=r.json()["data"]
    if not rows:raise ValueError("Empty FNG response")
    received=now_ms()
    path=root/"fng.parquet"
    # Historical availability is only an assumption. Lag by a full day.
    records=[{"event_time":int(x["timestamp"])*1000,"fng":int(x["value"]),
              "classification":x["value_classification"],"retrieved_ms":received,
              "available_ms":int(x["timestamp"])*1000+86400000,
              "availability_assumed":True} for x in rows]
    frame=pl.DataFrame(records)
    if not frame["fng"].is_between(0,100).all():raise ValueError("Invalid FNG range")
    if path.exists():frame=pl.concat([pl.read_parquet(path),frame])
    frame=frame.unique(subset=["event_time"],keep="first").sort("event_time")
    tmp=path.with_suffix(".tmp")
    frame.write_parquet(tmp);tmp.replace(path)
    print("FNG",frame.height,"daily rows",flush=True)

def score(db,revision):
    from transformers import pipeline
    model="ProsusAI/finbert"
    classifier=pipeline("text-classification",model=model,revision=revision,device=-1)
    while True:
        rows=db.execute("""SELECT id,headline FROM news WHERE NOT EXISTS
           (SELECT 1 FROM sentiment WHERE news_id=news.id AND model=? AND revision=?)
           LIMIT 32""",(model,revision)).fetchall()
        if not rows:break
        outputs=classifier([r[1] for r in rows],top_k=None,truncation=True,batch_size=8)
        for (key,_),out in zip(rows,outputs):
            scores={x["label"].lower():float(x["score"]) for x in out}
            db.execute("INSERT INTO sentiment VALUES (?,?,?,?,?,?,?)",
                (key,model,revision,now_ms(),scores["positive"],scores["negative"],scores["neutral"]))
        db.commit()
        print("Scored",len(rows),flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("command",choices=["fng","collect","watch","import","score"])
    p.add_argument("--root",type=Path,default=Path("data/external"))
    p.add_argument("--file",type=Path)
    p.add_argument("--interval",type=int,default=60)
    p.add_argument("--revision",help="Pinned FinBERT model commit SHA")
    a=p.parse_args()
    a.root.mkdir(parents=True,exist_ok=True)
    if a.command=="fng":fng(a.root);return
    with connect(a.root) as db:
        if a.command=="collect":
            if collect(db):raise SystemExit(1)
        elif a.command=="watch":
            if a.interval<30:p.error("interval must be >=30 seconds")
            while True:collect(db);time.sleep(a.interval)
        elif a.command=="import":
            if not a.file:p.error("--file required")
            import_jsonl(db,a.file)
        else:
            if not a.revision:p.error("--revision required; pin model weights")
            score(db,a.revision)
if __name__=="__main__":main()
