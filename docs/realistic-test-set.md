# A realistic test set: the plan

The two test sets (`scripts/cluster_urls.txt`, `scripts/holdout_urls.txt`) are small and clean: 258 of their 291 URLs are English Wikipedia pages, every page loads, and most topics have 5–6 pages. Real tabs look different. This set is meant to show how the tagging holds up on input like a real archive. It is a test set only: no prompt is tuned on it.

The plan is fixed before any page is gathered, and committed, so that a later change to it shows in git.

## Shape

About 500 pages, as one person's tabs might be:

| Part | Pages | What it tests |
|---|---|---|
| 7 big interests | ~245 | topics of 10–45 pages; related topics side by side |
| 14 medium topics | ~105 | topics of 5–10 pages |
| Long tail: 25 pairs and 100 single pages | ~150 | pages with no topic to share; tags shouldn't be forced on them |
| Ambiguous words | built in | one word for unrelated things, across topics |
| Other languages | ~40 | pages in 9 languages, labeled by topic like the rest |
| Pages that fail | ~30 | paywalls, sign-in walls, dead links, PDFs, apps that need JavaScript |
| Repeated URLs | ~10 extra lines | the same page with tracking parameters, a mobile or AMP address, a fragment |

Page sources, as real tabs have them: at most 1 page in 4 from Wikipedia (any language). The rest come from documentation, blogs and guides, forums and Q&A sites, product and shop pages, GitHub, news, and video or social posts.

## Topics

None of these topics, and none of the ambiguous words, is in the two current sets.

### Big interests

| Topic | Pages | Inside it | Languages |
|---|---|---|---|
| `apache-kafka` | 12 | docs, tutorials, Stack Overflow, a conference talk | |
| `apache-airflow` | 10 | docs, guides, Stack Overflow | |
| `apache-spark` | 10 | docs, Databricks posts (one on Delta Lake), Stack Overflow | |
| `postgresql` | 12 | docs (indexes, EXPLAIN, window functions), tuning posts | |
| `dbt` | 8 | docs, posts, GitHub | |
| `portugal` | 44 | a trip: Lisbon 15, Porto 10 (with port wine cellars), the Algarve 8, trains, airport and eSIMs 11 | 12 in Portuguese |
| `kitchen-renovation` | 40 | cabinets, countertops, appliances (with range hood airflow), budgets, contractors, permits; shop pages | |
| `mechanical-keyboards` | 28 | switches, builds and firmware, reviews and shops | 2 in Japanese |
| `spanish-language` | 26 | learning it: grammar, apps, podcasts, the DELE exam | 7 in Spanish |
| `woodworking` | 30 | joinery, hand tools (planes, chisels, sharpening), finishes, a workbench | 5 in German |
| `knee-rehab` | 12 | ACL and knee pain exercises, physiotherapy | |
| `strength-training` | 12 | squats, deadlifts, programs | |

Related topics side by side: the five data tools; knee rehab and strength training.

### Medium topics

| Topic | Pages | Languages |
|---|---|---|
| `chess` | 8 | 1 in French |
| `3d-printing` | 8 | |
| `aquariums` | 8 | |
| `home-networking` | 10 | |
| `home-automation` | 8 | 2 in German |
| `formula-1` | 8 | 1 in Spanish |
| `fermentation` | 8 | 2 in German, 1 in Korean |
| `franz-kafka` | 7 | 1 in German, 1 in Czech |
| `watercolor` | 7 | 1 in French |
| `urban-sketching` | 5 | |
| `react` | 8 | |
| `tabletop-rpg` | 7 | |
| `mushroom-foraging` | 7 | |
| `interview-prep` | 7 | |

Related topics side by side: home networking and home automation; watercolor and urban sketching.

### Long tail

**Pairs** (`pair-`): sous-vide, minimalism, tinnitus, standing-desks, solar-panels, wills-and-estates, cat-behavior, crosswords, vinyl-records, meal-prep, password-managers, vim, regex, unicode, calendar-history, probability-puzzles, color-theory, public-speaking, lawn-care, car-maintenance, river-deltas, nintendo-switch, airplanes, shipping-ports, sign-language.

**Single pages** (`one-`), 100:
- **Cooking (10):** shakshuka, storing fresh herbs, seasoning cast iron, why onions make you cry, tiramisu (in Italian), pho (in Vietnamese), sharpening kitchen knives, fresh pasta, cooking rice, egg substitutes in baking.
- **Science and health (14):** why the sky is blue, the placebo effect, how mRNA vaccines work, CRISPR, the Dunning–Kruger effect, the Mandela effect, migraine triggers, how much water to drink, blue-light glasses, how noise cancelling works, the Coriolis effect, how GPS works, lightning, the Mpemba effect.
- **Tech (16):** git rebase or merge (a Stack Overflow answer), find and xargs, JSON or YAML, how DNS works, OAuth 2.0, SSH keys for GitHub, falsehoods programmers believe about time, ripgrep, WebAssembly, how HTTPS works, semantic versioning, the twelve-factor app, when to use SQLite, the CAP theorem, a Big O cheat sheet, TCP or UDP.
- **History and culture (12):** the Hanseatic League, the Voynich manuscript, the Bauhaus, the Silk Road, the history of the QWERTY layout, the printing press, the Antikythera mechanism, Hokusai's Great Wave, the Rosetta Stone, the dancing plague of 1518, the Library of Alexandria, the Great Emu War.
- **Practical (12):** renewing a passport, a tax filing deadline, writing a cover letter, negotiating a salary, folding a fitted sheet, removing wine stains, unclogging a drain, tying a tie, a moving checklist, waste sorting in Germany (in German), getting rid of fruit flies, choosing a VPN.
- **News (8):** the EU AI Act, a Champions League final report, breaking at the Paris 2024 Olympics, congestion pricing in New York, a newly described species, housing prices in Spain (in Spanish), and two paywalled articles.
- **Entertainment (12):** a review of Dune: Part Two, Severance, a review of Project Hail Mary, a podcast episode, Hades, Stardew Valley, Hamilton, The Wire, Wordle, a Spotify playlist, a video essay on YouTube, a Netflix title page.
- **Places (8):** visiting the Rijksmuseum, Central Park, the Tour du Mont Blanc, Prague's old town, Iceland's Ring Road, Machu Picchu tickets, the Sagrada Família, Banff National Park.
- **Shopping (8):** a robot vacuum review, a backpack, a mattress buying guide, an air purifier review, a gift guide, a product page on Amazon, noise-cancelling headphones, electric toothbrushes.

Some of these sit near a topic on purpose: QWERTY history near keyboards, knife sharpening near woodworking, the Big O cheat sheet near interview prep, TCP and VPNs near home networking, color theory near watercolor.

### Ambiguous words

| Word | Meanings, by topic |
|---|---|
| kafka | `apache-kafka`, `franz-kafka` |
| switch | `mechanical-keyboards`, `home-networking`, `pair-nintendo-switch` |
| plane | `woodworking` (hand planes), `pair-airplanes` |
| port | `portugal` (port wine), `home-networking` (port forwarding), `pair-shipping-ports` |
| spark | `apache-spark`, `pair-car-maintenance` (spark plugs) |
| delta | `apache-spark` (Delta Lake), `pair-river-deltas` |
| airflow | `apache-airflow`, `kitchen-renovation` (range hoods) |

### Messy pages

These sit inside the topics above, as they would in real tabs:
- **Expected to fail, about 30:**
  - paywalled news (4);
  - Reddit threads (6);
  - YouTube videos (5);
  - posts on X (2);
  - a shared Google Doc and a Notion page;
  - dead links on live sites (3);
  - PDFs (3);
  - apps that need JavaScript (3: a chess board, a map, a virtual tabletop);
  - shops that block robots (3).
- **Repeated pages, about 10:** the same page with `utm_` parameters, a mobile Wikipedia address, an AMP address, a fragment, `http://`. The import should fold each into the page it repeats.

## File format

`scripts/realistic_urls.txt`, in the format of `holdout_urls.txt`, so the import can read it as it is:
- `# cluster: name` starts a topic. Single pages are `one-` topics, pairs `pair-` topics.
- `# ambiguous: a b` names topics that share a word.
- `# group: name: topic topic …` names related topics, for a coarser second score.
- `# fails: URL reason` marks a page expected to fail.
- `# repeats:` starts the repeated URLs, which belong to no topic.

The fetched pages are kept in a page cache under `target/experiments/`, so experiments don't fetch them again.

## Scoring

`scripts/diagnose.py`, with additions:
- Pages that failed to load are left out of F1 and the untagged counts, and reported on their own, with whether the failure was expected.
- Repeated URLs: how many the import folded.
- Groups: F1 over the `# group:` topics, as a coarser view.
- Languages: how often a page not in English gets its topic's tag first, against English pages. The language is the one the import detected.
- Ambiguous words: collisions, as for the held-out set.

## How the pages are gathered

- For each topic, web searches and the sites' own indexes: Wikipedia's API (category members, the same article in other languages), documentation tables of contents. No URL is typed from memory.
- The first result that fits the plan is taken, not the one most likely to tag well.
- Every page is fetched once and checked: it loads (or fails as the plan says), and it is about its topic. Pages that don't fit are replaced the same way.
- Then two imports and their reviews give the baseline. Before that, the topic labels are checked against the pages' titles and summaries.

Expected time:
- gathering and checking: 1–2 hours;
- each import: about 40 minutes;
- each review: about 15 minutes.

## Limits

It is still a guess at what a tab archive looks like, by the same author as the prompts, and it will be cleaner than real tabs. A later check against 50–100 real tabs would show how representative it is. Adding another 500 pages later would double as a real growth test (section 11.6 of `tagging-algorithm.md`).
