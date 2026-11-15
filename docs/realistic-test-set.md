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

## As gathered

`scripts/realistic_urls.txt` has 501 pages in 152 topics:
- 26 topics of 5–44 pages;
- 25 pairs;
- 101 single pages;
- 13 repeated URLs.

The pages come from 334 domains:
- 93 from Wikipedia (19%);
- 22 documentation pages;
- 18 Stack Overflow and Stack Exchange pages;
- 13 GitHub repositories;
- 344 blogs, guides, news, shops and other sites;
- 40 pages in 9 languages besides English.

The pages were fetched once with tabkeeper's own fetcher, through an import whose model was a stand-in, into a page cache (`target/experiments/r500/pages`). Where the result differs from the plan:
- **53 pages are expected to fail, not about 30.**
  - Stack Overflow answers every request from tabkeeper's fetcher with 403. The fetcher names itself as a bot, and several other sites refuse it or rate-limit it too (403, 429, 418, dropped connections).
  - These pages stay in the set, marked `# fails:`, since real tabs would fail the same way.
  - Three dead links stay, as planned.
  - The other 23 dead links (old pages linked from Hacker News) were replaced by the next fitting result.
- **YouTube, Reddit, Notion, Spotify, Google Maps and paywalled pages don't fail: they load thin.** tabkeeper counts them as loaded, with a title and little else. The 20 pages that load with under 500 characters of text are marked `# thin:`.
- **The import folds 5 of the 13 repeated URLs**: tracking parameters, fragments and a trailing slash. It keeps the other 8 as pages of their own:
  - `http://` and `https://` forms of one address;
  - `www.` and the bare domain;
  - mobile Wikipedia;
  - AMP and print views;
  - a country path;
  - a second address for one post.

## Baseline

Two imports of the set with the code of 95cce6f, each reviewed once at temperature 0. An import took 32 minutes and a review 10–11. The scores are for the 314 loaded pages on the 26 topics of 5–44 pages. The other two sets' numbers are from section 12 of `tagging-algorithm.md`.

| | This set | Held-out set | Old set |
|---|---|---|---|
| First-tag pairwise F1 | **0.47** (0.51, 0.44) | 0.965 (0.97, 0.96) | 0.905 (0.88, 0.93) |
| Precision | 0.97 (0.99, 0.95) | 0.99 (0.98, 1.00) | 0.97 (0.95, 0.99) |
| Recall | **0.31** (0.34, 0.28) | 0.95 (0.97, 0.93) | 0.855 (0.83, 0.88) |
| First tag from the page's own topic | 95% (301, 296) | 97% (109, 107 of 111) | 95% (134, 134 of 141) |
| Pages untagged | 10, 9 | 1, 4 | 3, 6 |
| Topics lost | 0 | 0 | 0 |
| Coverage, all pages | 78% (79%, 77%) | 76% | 97% |
| Two imports, each reviewed, agree on | **0.70** | 0.92 | 0.91 |
| Collisions on ambiguous words | 0 | 0 | 0–1 |

On this set only:
- **Other languages:** pages in other languages got a first tag from their own topic as often as English pages: 66 of 68 (34, 32 of 34), against 531 of 560.
- **Thin pages:** in each review, 8 of the 11 on these topics got their topic's tag, and 2 were left untagged.
- **Pages that didn't load:** 53, all of them expected to fail.
- **The long tail:**
  - 51 and 39 of its 134 loaded pages got a tag.
  - A topic needs 3 pages for a tag of its own (`min_support`), so a pair shares a first tag only by chance: 4 of 21 pairs in each review.
- **Without the review,** the imports score F1 0.14, since each page gets narrow tags of its own.

### What holds

- **Topics stay apart** as well as on the other two sets, and no topic is lost.
- **Other languages:** pages in the 9 other languages are tagged as well as English ones.
- **Ambiguous words:** none of the seven puts two meanings under one first tag.
- **Failures:** every page that failed was expected to.

### What goes wrong

1. **Big topics are split into subtopics, with no tag over them.**
   - Precision is as on the other sets, but recall drops to 0.31. Topics of 25–40 pages get 3–6 tags each. In the first review:
     - `portugal` (39 loaded pages): lisbon-travel 12, algarve 8, portugal-travel 7, portugal-trains 5, portugal 4, lisbon 3;
     - `kitchen-renovation` (35): kitchen-remodeling 11, countertops 7, kitchen-ventilation 5, kitchen-cabinets 4, induction-cooking 4, others 4;
     - `spanish-language` (25): spanish-language-learning 9, spanish-subjunctive 5, spanish-grammar 5, spanish-language-certification 4, spanish-verbs 2.
   - Topics of 5–10 pages mostly keep one tag.
   - Nothing joins the parts:
     - Pages have 1.4–1.5 tags on average.
     - The one tag that finds most of a topic's pages finds 20–23 of the 39 Portugal pages, and 13–14 of the 35 kitchen pages.
     - The tree doesn't join them either: the first review's tree files `lisbon` and `portugal` under culture and the other Portugal tags under travel.
   - The cause: tags are named per cluster, and the clustering joins pages only while they are alike enough (`cluster_similarity`). A broad interest's pages aren't all that alike, so it becomes several clusters, each with its own name.
2. **Near-duplicate tags share one vocabulary.**
   - Examples:
     - `mechanical-keyboard` and `mechanical-keyboards`, beside `keyboards`;
     - `lisbon` and `lisbon-travel`;
     - `portugal` and `portugal-travel`;
     - `nitrogen-cycle` and `aquarium-cycling`;
     - `firmware` and `keyboard-firmware`.
   - The vocabulary review has no merge step:
     - with `reconcile.vocabulary` on (the default), the older review's merge check doesn't run;
     - a cluster's name reuses a tag only when it is spelled the same.
   - Merging the clear duplicates by hand raises F1 to just 0.55 and 0.49. So most of the split is real subtopics: Lisbon, the Algarve, trains.
3. **Two runs split differently.** Two imports of the set, each reviewed, agree on 0.70. Where a big topic splits changes from run to run.
4. **Loose catch-all tags in the long tail.** Most long-tail pages stay untagged, as planned, but some tags gather unrelated pages:
   - `minimalism` (first review): two posts on minimalism in software, plus Vim, the twelve-factor app, semantic versioning, color theory, furniture design software, a Nintendo Switch page and two essays on woodworking;
   - `time` (second review): leap seconds and the Gregorian calendar, but also GPS, the Antikythera mechanism and Apple's sign-language service, SignTime;
   - `puzzles` (first review): crosswords and the Monty Hall problem, and also 85 ways to tie a tie;
   - `portugal-travel` (second review) on a Google Maps link and on the Tour du Mont Blanc.
5. **The data tools share generic tags.**
   - `apache-spark` is almost all `data-processing`.
   - `dbt` splits over data-transformation, data-engineering and data-pipelines, and those tags reach Airflow and PostgreSQL pages too.
   - Most pages with another topic's first tag are these.
6. **One naming reply ran past the 1,500-token cap,** in the first review.
   - It was for a cluster of three unrelated pages: two thin Reddit threads and a page on Kafka's insurance job.
   - Its pages were left for the next review.
   - The longest reply seen before was about 450 tokens.

## Limits

It is still a guess at what a tab archive looks like, by the same author as the prompts, and it will be cleaner than real tabs. A few pages fit their topic loosely: one meal-prep page is an article on food flown to a village in Alaska. A later check against 50–100 real tabs would show how representative it is. Adding another 500 pages later would double as a real growth test (section 11.6 of `tagging-algorithm.md`).
