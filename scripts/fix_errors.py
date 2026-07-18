# scripts/fix_errors.py
import json
from pathlib import Path
from datetime import datetime

LABELS_DIR = Path("datasets/labels")

# Các file lỗi từ check_data output
ERROR_FILES = [
    "mcocr_public_145013jcwbe.json",
    "mcocr_public_145013nfkiq.json",
    "mcocr_public_145013qiyft.json",
    "mcocr_public_145013satyz.json",
    "mcocr_public_145013sbulh.json",
    "mcocr_public_145013tffpu.json",
    "mcocr_public_145014aclha.json",
    "mcocr_public_145014ansoe.json",
    "mcocr_public_145014jkegi.json",
    "mcocr_public_145014mdhan.json",
    "mcocr_public_145014mfmvh.json",
    "mcocr_public_145014mppzy.json",
    "mcocr_public_145014mpsud.json",
    "mcocr_public_145014rqund.json",
    "mcocr_public_145014shmit.json",
    "mcocr_public_145014uqaia.json",
    "mcocr_public_145014woqkn.json",
    "mcocr_public_145014wpplc.json",
    "mcocr_public_145014xbkfz.json",
    "mcocr_public_145014xzqlh.json",
    "mcocr_public_145014yrxvd.json",
    "mcocr_public_145014yxijr.json",
]


def try_fix_date(v: str) -> str | None:
    """Thử parse nhiều format date, trả về YYYY-MM-DD hoặc None."""
    formats = [
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%Y/%m/%d",
        "%d/%m/%y",
        "%Y%m%d",
    ]
    # Bỏ ký tự thừa
    v = v.strip().replace(".", "/")
    # Nếu bị cắt ngắn (16/08/202) → không fix được
    for fmt in formats:
        try:
            return datetime.strptime(v, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


auto_fixed = []
need_manual = []

for fname in ERROR_FILES:
    fpath = LABELS_DIR / fname
    if not fpath.exists():
        need_manual.append((fname, "FILE NOT FOUND"))
        continue

    data = json.loads(fpath.read_text(encoding="utf-8"))
    issues = []
    fixed = []

    # ── Fix 1: items là None → đổi thành [] ──────────────────────────
    if data.get("items") is None:
        data["items"] = []
        fixed.append("items: null → []")

    # ── Fix 2: total là None → thử tính từ items ─────────────────────
    if data.get("total") is None or data.get("total") == 0:
        if data["items"]:
            calc = sum(float(i.get("total_price", 0) or 0) for i in data["items"])
            if calc > 0:
                data["total"] = calc
                fixed.append(f"total: null → {calc} (sum of items)")
            else:
                issues.append("total=0 và items rỗng → cần điền tay")
        else:
            issues.append("total=null, items rỗng → cần điền tay")

    # ── Fix 3: store_name là None hoặc "" ─────────────────────────────
    if not data.get("store_name"):
        data["store_name"] = "UNKNOWN"
        fixed.append("store_name: null → 'UNKNOWN' (cần sửa tay)")
        issues.append("store_name='UNKNOWN' — cần xem ảnh và điền đúng")

    # ── Fix 4: date format sai ────────────────────────────────────────
    date_val = data.get("date", "")
    if date_val:
        try:
            datetime.strptime(date_val, "%Y-%m-%d")  # đã đúng
        except ValueError:
            fixed_date = try_fix_date(date_val)
            if fixed_date:
                data["date"] = fixed_date
                fixed.append(f"date: '{date_val}' → '{fixed_date}'")
            else:
                issues.append(f"date='{date_val}' không parse được → cần điền tay")

    # ── Ghi lại nếu có fix ───────────────────────────────────────────
    if fixed:
        fpath.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    if issues:
        need_manual.append((fname, "; ".join(issues), fixed))
    else:
        auto_fixed.append((fname, fixed))

# ── Report ────────────────────────────────────────────────────────────
print(f"\n{'=' * 55}")
print(f"  Auto-fixed: {len(auto_fixed)}")
print(f"  Need manual: {len(need_manual)}")
print(f"{'=' * 55}")

if auto_fixed:
    print("\n✅ Auto-fixed:")
    for fname, fixes in auto_fixed:
        print(f"  {fname}")
        for f in fixes:
            print(f"    → {f}")

if need_manual:
    print("\n⚠️  Cần fix tay:")
    for item in need_manual:
        fname, issue = item[0], item[1]
        fixes = item[2] if len(item) > 2 else []
        print(f"  {fname}")
        if fixes:
            print(f"    (đã auto-fix: {', '.join(fixes)})")
        print(f"    ❌ {issue}")
