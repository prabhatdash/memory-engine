"""Deterministic generator for the multi-session persona benchmark (PMB).

Every scenario is scripted from templates with a fixed seed so that ground
truth can never drift from what the "user" actually said (the contamination
problem reported in the draft paper's single LLM-simulated run).

Per persona:
  Session 1: 8 fact statements + 2 distractor facts about other people + fillers
  Session 2: 3 value updates + 2 paraphrased restatements + fillers
  Session 3: probes (fresh session), each asked independently:
      5 recall probes   (facts never updated)
      3 update probes   (must return the NEW value; returning the old one = stale)
      2 abstention probes (facts never stated; must say "don't know")
"""
import json, random, sys

SLOTS = {
    "city": dict(
        values=["Tromso", "Ljubljana", "Valparaiso", "Hobart", "Cluj-Napoca", "Tbilisi", "Halifax", "Dunedin"],
        state=["I recently settled in {v}, and honestly I love the vibe there.",
               "These days I live in {v}; the rent is fine and the coffee is great."],
        restate="Like I said before, life in {v} is treating me well.",
        update="Big news: I've just relocated from {old} to {v} for work.",
        probe="Which city am I living in at the moment?",
        distract="My cousin Arjun lives in {v}.",
        summary=["The user lives in {v}.", "The user is based in the city of {v}.", "The user's home city is {v}."],
        updatable=True),
    "budget": dict(
        values=["1.5 lakh rupees", "3,200 euros", "4,700 dollars", "2.8 lakh rupees", "6,100 euros", "9,400 dollars"],
        state=["My total budget for the side project is {v}.", "I've set aside {v} as the budget for my side project."],
        restate="Just to remind you, the side-project budget is {v}.",
        update="Good news: I secured extra funding, so the side-project budget went from {old} to {v}.",
        probe="How much budget do I have for my side project?",
        distract="My colleague Marco's startup budget is {v}.",
        summary=["The user's side-project budget is {v}.", "The user has {v} available for the side project.", "The budget the user set for the side project is {v}."],
        updatable=True),
    "framework": dict(
        values=["Svelte", "Vue", "Angular", "SolidJS", "Qwik", "Ember"],
        state=["For my app's frontend I've decided to use {v}.", "I'm building the frontend of my app with {v}."],
        restate="Anyway, as you know I'm doing the frontend in {v}.",
        update="I changed my mind: I'm switching the app's frontend from {old} to {v}.",
        probe="Which frontend framework am I using for my app?",
        distract="My friend Lena builds all her apps with {v}.",
        summary=["The user is using {v} for the app frontend.", "The user's app frontend is built with {v}.", "The user chose {v} as the frontend framework."],
        updatable=True),
    "hours": dict(
        values=["6", "9", "14", "22", "11", "17"],
        state=["I can only spare {v} hours a week for the project.", "Realistically I have {v} hours per week to work on the project."],
        restate="Remember, I only have {v} hours a week for this.",
        update="My schedule changed: I can now give the project {v} hours per week instead of {old}.",
        probe="How many hours per week can I spend on my project?",
        distract="My brother-in-law works {v} hours a week at the hospital.",
        summary=["The user can spend {v} hours per week on the project.", "The user has {v} weekly hours available for the project.", "The user's weekly time budget for the project is {v} hours."],
        updatable=True),
    "diet": dict(
        values=["vegetarian", "vegan", "pescatarian", "keto", "gluten-free"],
        state=["Just so you know, I follow a {v} diet.", "I eat a strictly {v} diet these days."],
        restate="As a {v} eater, I'm always hunting for recipes.",
        update="I've switched from a {old} diet to a {v} diet.",
        probe="What kind of diet do I follow?",
        distract="My roommate Priya follows a {v} diet.",
        summary=["The user follows a {v} diet.", "The user eats {v}.", "The user's diet is {v}."],
        updatable=True),
    "job": dict(
        values=["marine biologist", "tax auditor", "bakery owner", "air traffic controller", "museum curator", "sound engineer"],
        state=["I work as a {v}.", "By profession I'm a {v}."],
        restate="Work as a {v} keeps me busy, as you know.",
        update="Career update: I quit being a {old} and started a new job as a {v}.",
        probe="What do I do for work?",
        distract="My neighbour Tom works as a {v}.",
        summary=["The user works as a {v}.", "The user's profession is {v}.", "The user is employed as a {v}."],
        updatable=True),
    "trip": dict(
        values=["Patagonia", "Kyoto", "Reykjavik", "Zanzibar", "Lofoten", "Cappadocia"],
        state=["I'm planning a trip to {v} this spring.", "My big spring trip this year is to {v}."],
        restate="I'm still excited about the spring trip to {v}.",
        update="Change of plans: the spring trip is now to {v} instead of {old}.",
        probe="Where am I travelling this spring?",
        distract="My aunt Rosa is travelling to {v} this spring.",
        summary=["The user is planning a spring trip to {v}.", "The user will travel to {v} in spring.", "The user's spring travel destination is {v}."],
        updatable=True),
    "pet": dict(
        values=["Biscotti", "Waffles", "Pistachio", "Noodle", "Mochi", "Pickles", "Ziggy", "Marmalade"],
        state=["My dog {v} chewed through my headphone cable again today.", "I have a dog called {v} who is very needy."],
        restate="{v}, my dog, says hi by the way.",
        probe="What is my dog's name?",
        distract="My neighbour's dog is called {v}.",
        summary=["The user has a dog named {v}.", "The user's dog is called {v}.", "{v} is the name of the user's dog."],
        updatable=False),
    "sibling": dict(
        values=["Anushka", "Freya", "Ingrid", "Yasmin", "Leonie", "Tamsin"],
        state=["My sister {v} is visiting next month.", "I'm close to my sister {v}; we talk every day."],
        restate="My sister {v} keeps asking about my project.",
        probe="What's my sister's name?",
        distract="My best friend's sister is called {v}.",
        summary=["The user's sister is named {v}.", "The user has a sister called {v}.", "{v} is the user's sister."],
        updatable=False),
    "allergy": dict(
        values=["peanuts", "shellfish", "penicillin", "kiwi", "sesame", "latex"],
        state=["I'm seriously allergic to {v}, so I have to be careful.", "Important: I have a severe {v} allergy."],
        restate="Because of my {v} allergy I always check labels.",
        probe="What am I allergic to?",
        distract="My colleague Marco is allergic to {v}.",
        summary=["The user is allergic to {v}.", "The user has a severe {v} allergy.", "The user must avoid {v} due to an allergy."],
        updatable=False),
    "instrument": dict(
        values=["cello", "banjo", "theremin", "oboe", "ukulele", "harp"],
        state=["I've been learning the {v} in the evenings.", "My new hobby is playing the {v}."],
        restate="Practising the {v} is going better lately.",
        probe="Which musical instrument am I learning?",
        distract="My nephew is learning the {v}.",
        summary=["The user is learning to play the {v}.", "The user's hobby is playing the {v}.", "The user practises the {v} in the evenings."],
        updatable=False),
}

FILLERS = [
    "Can you explain recursion in two sentences?",
    "Give me one quick tip for better sleep.",
    "What's the difference between TCP and UDP, briefly?",
    "Suggest a catchy name for a productivity newsletter.",
    "How do I keep basil plants alive indoors?",
    "What is a good stretch for lower back pain?",
    "Summarise the idea of compound interest in one paragraph.",
    "Write a haiku about Monday mornings.",
    "What are three tips for writing a clear email?",
    "How long should I boil an egg for a jammy yolk?",
    "Explain what an API is to a ten-year-old.",
    "Give me a 3-item checklist before a job interview.",
    "What's the capital of Australia, and why isn't it Sydney?",
    "Recommend a simple warm-up before running.",
    "How does a bloom filter work, in short?",
    "What is the Pomodoro technique?",
    "Give me a quick idea for a weeknight dinner.",
    "What does 'idempotent' mean in HTTP?",
    "How can I stop procrastinating on boring tasks?",
    "Tell me a fun fact about octopuses.",
    "What's a polite way to decline a meeting?",
    "Explain the difference between RAM and storage.",
    "Give me two ideas for a rainy weekend.",
    "What is the 80/20 rule?",
    "How do I make my git commit messages better?",
    "What is a good way to learn touch typing?",
    "Explain big-O notation in one paragraph.",
    "How do noise-cancelling headphones work?",
    "Give me a motivational quote and explain it.",
    "What are the benefits of journaling?",
]

ABSTAIN_PROBES = {k: v["probe"] for k, v in SLOTS.items()}


def build(n_personas=12, seed=7, fillers_s1=14, fillers_s2=8):
    rng = random.Random(seed)
    updatable = [k for k, v in SLOTS.items() if v["updatable"]]
    fixed = [k for k, v in SLOTS.items() if not v["updatable"]]
    scenarios = []
    for p in range(n_personas):
        # 3 updated + 5 recall (non-updated) slots, 2 unstated slots for abstention
        upd = rng.sample(updatable, 3)
        rest = [k for k in SLOTS if k not in upd]
        rng.shuffle(rest)
        recall, abstain = rest[:5], rest[5:7]
        values, turns_s1, turns_s2 = {}, [], []
        for slot in upd + recall:
            v = rng.choice(SLOTS[slot]["values"])
            values[slot] = {"value": v}
            turns_s1.append({"kind": "fact", "slot": slot, "text": rng.choice(SLOTS[slot]["state"]).format(v=v)})
        for slot in upd:
            old = values[slot]["value"]
            new = rng.choice([x for x in SLOTS[slot]["values"] if x != old])
            values[slot].update(old=old, value=new)
            turns_s2.append({"kind": "update", "slot": slot, "text": SLOTS[slot]["update"].format(v=new, old=old)})
        # distractors: other people's facts with a different value, on recall slots
        for slot in rng.sample(recall, 2):
            dv = rng.choice([x for x in SLOTS[slot]["values"] if x != values[slot]["value"]])
            values[slot]["distractor"] = dv
            turns_s1.append({"kind": "distractor", "slot": slot, "text": SLOTS[slot]["distract"].format(v=dv)})
        # restatements of two recall facts in session 2 (duplicate-handling test)
        for slot in rng.sample(recall, 2):
            turns_s2.append({"kind": "restate", "slot": slot, "text": SLOTS[slot]["restate"].format(v=values[slot]["value"])})
        fill = rng.sample(FILLERS, fillers_s1 + fillers_s2)
        turns_s1 += [{"kind": "filler", "text": t} for t in fill[:fillers_s1]]
        turns_s2 += [{"kind": "filler", "text": t} for t in fill[fillers_s1:]]
        # facts must precede their updates, otherwise order is random
        rng.shuffle(turns_s1)
        rng.shuffle(turns_s2)
        probes = []
        for slot in recall:
            probes.append({"type": "recall", "slot": slot, "q": SLOTS[slot]["probe"],
                           "answer": values[slot]["value"], "distractor": values[slot].get("distractor")})
        for slot in upd:
            probes.append({"type": "update", "slot": slot, "q": SLOTS[slot]["probe"],
                           "answer": values[slot]["value"], "stale": values[slot]["old"]})
        for slot in abstain:
            probes.append({"type": "abstain", "slot": slot, "q": SLOTS[slot]["probe"]})
        scenarios.append({"id": f"P{p+1:02d}", "values": values,
                          "sessions": [turns_s1, turns_s2], "probes": probes})
    return scenarios


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    out = sys.argv[2] if len(sys.argv) > 2 else "data/scenarios.json"
    sc = build(n)
    json.dump({"slots": {k: {"summary": v["summary"], "values": v["values"], "probe": v["probe"]} for k, v in SLOTS.items()},
               "scenarios": sc}, open(out, "w"), indent=1)
    t = sum(len(s["sessions"][0]) + len(s["sessions"][1]) for s in sc)
    pr = sum(len(s["probes"]) for s in sc)
    print(f"{len(sc)} personas, {t} ingestion turns, {pr} probes -> {out}")
