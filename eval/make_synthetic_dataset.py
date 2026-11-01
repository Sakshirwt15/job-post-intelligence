"""50 labelled posts ka STARTER set generate karta hai (messy Hinglish templates, known ground truth).

Ye synthetic hai - resume me 'real' claim karne ke liye apne Telegram groups ki posts se
eval/labelled_posts.json me kam se kam 20-30 real posts replace/add karo (README dekho).

Run:  python -m eval.make_synthetic_dataset
"""
import json
import random
from pathlib import Path

OUT = Path(__file__).parent / "labelled_posts.json"
rng = random.Random(42)

COMPANIES = ["NovaStack Technologies", "Bluepeak Labs", "QuantaWorks", "Zenith Analytics", "PixelForge Studios",
             "CloudNimbus", "DataBridge AI", "FinEdge Payments", "GreenLeaf Robotics", "Orbit Commerce",
             "Lumina Health", "TerraByte Systems", "Vantage Retail Tech", "Kiteflow", "Mintra Fintech"]
ROLES = [("AI/ML Intern", ["Python", "PyTorch", "NLP"], "internship"),
         ("Data Analyst Intern", ["SQL", "Excel", "Power BI"], "internship"),
         ("Backend Developer", ["Python", "FastAPI", "PostgreSQL"], "full-time"),
         ("Frontend Developer Intern", ["React", "JavaScript", "CSS"], "internship"),
         ("Full Stack Developer", ["Node.js", "React", "MongoDB"], "full-time"),
         ("DevOps Engineer", ["Docker", "Kubernetes", "AWS"], "full-time"),
         ("Data Scientist", ["Python", "Machine Learning", "SQL"], "full-time"),
         ("Android Developer Intern", ["Kotlin", "Android", "Firebase"], "internship"),
         ("Software Engineer", ["Java", "Spring Boot", "SQL"], "full-time"),
         ("Generative AI Intern", ["LangChain", "Python", "LLMs"], "internship"),
         ("QA Automation Intern", ["Selenium", "Python", "Pytest"], "internship"),
         ("Cloud Engineer", ["AWS", "Terraform", "Linux"], "full-time")]
CITIES = ["Pune", "Bangalore", "Mumbai", "Hyderabad", "Delhi", "Chennai", "Gurgaon", "Noida"]
BATCHES = [[2025], [2026], [2025, 2026], [2024, 2025], [2026, 2027]]
INTERN_PAY = [(10000, 10000), (15000, 15000), (20000, 20000), (25000, 25000), (30000, 30000), (15000, 20000),
              (20000, 30000), (25000, 40000), (None, None)]
LPA = [3, 6, 9, 12]


def fmt_k(n):
    return f"{n // 1000}k"


def pay_text(lo, hi, kind, variant):
    if lo is None:
        return rng.choice(["Stipend: as per interview", "Unpaid initially, PPO possible", "Stipend discussed in interview"])
    if kind == "lpa":
        return rng.choice([f"{lo * 12 / 100000:g} LPA", f"CTC: {lo * 12 / 100000:g} LPA", f"Salary {lo * 12 / 100000:g} lpa"])
    if lo == hi:
        return [f"₹{lo:,}/month", f"{fmt_k(lo)} per month", f"Stipend: {lo} pm", f"{fmt_k(lo)} stipend"][variant % 4]
    return [f"₹{lo:,} - ₹{hi:,}/month", f"{fmt_k(lo)}-{fmt_k(hi)}/month", f"{lo // 1000}-{hi // 1000}k pm",
            f"Stipend: {fmt_k(lo)} to {fmt_k(hi)}"][variant % 4]


def batch_text(b, variant):
    if len(b) == 1:
        return [f"{b[0]} batch", f"Batch {b[0]}", f"{b[0]} passouts only", f"{b[0]} graduates"][variant % 4]
    j = " & ".join(map(str, b))
    s = "/".join(map(str, b))
    return [f"{s} batches", f"{j} passouts", f"Batch: {s}", f"{j} pass-outs welcome"][variant % 4]


def make_link(i, company):
    slug = "".join(c for c in company.lower() if c.isalnum())
    return rng.choice([f"https://careers.{slug}.com/apply/{1000 + i}", f"https://forms.gle/Xy{i:03d}Qz",
                       f"https://{slug}.in/jobs/{i}", f"https://bit.ly/job{i:03d}"])


TEMPLATES = [
    lambda f: f"🚀 HIRING ALERT 🚀\n{f['company']} is looking for {f['role']}\n📍 {f['loc_t']}\n💰 {f['pay_t']}\n🎓 {f['batch_t']}\n🛠 Skills: {', '.join(f['skills'])}\n👉 Apply: {f['contact']}\n\nShare with friends!",
    lambda f: f"Hey guys! {f['company']} me opening hai - {f['role']}. Location: {f['loc_t']}. {f['pay_t']}. {f['batch_t']} eligible hain. Aana chahiye: {' | '.join(f['skills'])}. Apply karo jaldi: {f['contact']} 🔥",
    lambda f: f"*{f['company']}*\nRole: {f['role']}\nLocation: {f['loc_t']}\nCompensation: {f['pay_t']}\nEligibility: {f['batch_t']}\nTech: {', '.join(f['skills'])}\nHow to apply: {f['contact']}",
    lambda f: f"Forwarded as received\n\nBhai referral chahiye to ping karo. {f['company']} hiring {f['role']} ({f['loc_t']}), {f['pay_t']}. {f['batch_t']}. Must know {', '.join(f['skills'][:-1])} and {f['skills'][-1]}. {f['contact']}",
    lambda f: f"📢 Opportunity @ {f['company']}\n\nWe're hiring a {f['role']}!\n• Where: {f['loc_t']}\n• Pay: {f['pay_t']}\n• Who: {f['batch_t']}\n• Stack: {', '.join(f['skills'])}\n\nInterested? {f['contact']}\n#hiring #freshers #jobs",
    lambda f: f"{f['role']} - {f['company']} ({f['loc_t']}) | {f['pay_t']} | {f['batch_t']} | {'/'.join(f['skills'])} | {f['contact']}",
]


def build_post(i, company, role, skills, emp, city, remote, batch, pay, kind, link, email, template_idx, variant):
    contact = email_t = None
    if email:
        contact = f"mail resume to {email}"
    else:
        contact = link
    loc_t = "Remote / WFH" if remote else (f"{city} (onsite)" if variant % 2 else city)
    f = {"company": company, "role": role, "skills": skills, "loc_t": loc_t,
         "pay_t": pay_text(pay[0], pay[1], kind, variant), "batch_t": batch_text(batch, variant), "contact": contact}
    return TEMPLATES[template_idx](f)


def generate(n_unique=44, n_repost=6):
    items = []
    for i in range(n_unique):
        company = COMPANIES[i % len(COMPANIES)] if i < len(COMPANIES) else rng.choice(COMPANIES)
        role, skills, emp = ROLES[(i * 5 + rng.randint(0, 3)) % len(ROLES)]
        remote = rng.random() < 0.15
        city = None if remote else rng.choice(CITIES)
        batch = rng.choice(BATCHES)
        if emp == "internship":
            pay = rng.choice(INTERN_PAY)
            kind = "monthly"
        else:
            lpa = rng.choice(LPA)
            pay, kind = (lpa * 100000 // 12,) * 2, "lpa"
        use_email = rng.random() < 0.12
        link = None if use_email else make_link(i, company)
        email = f"careers@{''.join(c for c in company.lower() if c.isalnum())}.com" if use_email else None
        items.append(dict(i=i, company=company, role=role, skills=skills, emp=emp, city=city, remote=remote,
                          batch=batch, pay=pay, kind=kind, link=link, email=email))

    posts = []
    for k, it in enumerate(items):
        text = build_post(it["i"], it["company"], it["role"], it["skills"], it["emp"], it["city"], it["remote"],
                          it["batch"], it["pay"], it["kind"], it["link"], it["email"], k % len(TEMPLATES), k)
        posts.append({"id": len(posts) + 1, "group_id": k + 1, "text": text, "labels": label(it)})
    for r in range(n_repost):  # reposts: same job, alag template/wording
        it = items[r * 7 % n_unique]
        text = build_post(it["i"], it["company"], it["role"], it["skills"], it["emp"], it["city"], it["remote"],
                          it["batch"], it["pay"], it["kind"], it["link"], it["email"],
                          (r * 7 % n_unique + 3) % len(TEMPLATES), r * 7 % n_unique + 1)
        posts.append({"id": len(posts) + 1, "group_id": r * 7 % n_unique + 1, "text": text, "labels": label(it)})
    return posts


def label(it):
    lo, hi = it["pay"]
    return {"company": it["company"], "role": it["role"], "employment_type": it["emp"], "batch": it["batch"],
            "stipend_min": lo, "stipend_max": hi, "location": it["city"], "is_remote": it["remote"],
            "skills": it["skills"], "apply_link": it["link"], "apply_email": it["email"]}


if __name__ == "__main__":
    data = generate()
    OUT.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(data)} labelled posts -> {OUT}")
