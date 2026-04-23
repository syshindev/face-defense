import argparse
import os
import sys
import time

TPDNE_URL = "https://thispersondoesnotexist.com/image"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "image/avif,image/webp,image/*,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://thispersondoesnotexist.com/",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Download N StyleGAN faces from thispersondoesnotexist")
    parser.add_argument("--out_dir", type=str, default="data/extra/stylegan")
    parser.add_argument("--count", type=int, default=10000)
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Delay between requests (seconds)")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Starting filename index (for resuming)")
    parser.add_argument("--use_cloudscraper", action="store_true",
                        help="Use cloudscraper if plain requests fails")
    return parser.parse_args()


def get_session(use_cloudscraper):
    if use_cloudscraper:
        import cloudscraper
        return cloudscraper.create_scraper()
    import requests
    sess = requests.Session()
    sess.headers.update(HEADERS)
    return sess


def download_one(sess, path):
    r = sess.get(TPDNE_URL, timeout=30)
    if r.status_code != 200 or len(r.content) < 10_000:
        return False, f"status={r.status_code} size={len(r.content)}"
    with open(path, "wb") as f:
        f.write(r.content)
    return True, f"size={len(r.content)}"


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    sess = get_session(args.use_cloudscraper)

    saved = 0
    failed = 0
    start = time.time()
    for i in range(args.count):
        idx = args.start_idx + i
        path = os.path.join(args.out_dir, f"stylegan_{idx:06d}.jpg")
        if os.path.exists(path):
            saved += 1
            continue
        ok, msg = download_one(sess, path)
        if ok:
            saved += 1
        else:
            failed += 1
            print(f"  [{idx}] FAIL {msg}", flush=True)
            if failed > 20:
                print("Too many failures, aborting. Try --use_cloudscraper.",
                      file=sys.stderr)
                sys.exit(1)
        if (saved + failed) % 50 == 0:
            elapsed = time.time() - start
            rate = (saved + failed) / max(elapsed, 1)
            print(f"  saved={saved} failed={failed} "
                  f"elapsed={elapsed:.0f}s rate={rate:.1f}/s", flush=True)
        time.sleep(args.delay)

    total = time.time() - start
    print(f"\nDone. saved={saved} failed={failed} total={total:.0f}s")


if __name__ == "__main__":
    main()
