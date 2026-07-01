# eval run — 2026-07-01-research-hedgepolicy-true-answers

10 cases, 7 passed all applicable checks. Each section is one case (structural header + answer body).

---

## bio-euclid

- model: `audrey_research`
- status: PASS
- route: research
- latency: route:research  ttft:10.6s  total:576.8s
- banners: Planning → Researching → Verifying → Fact-checking → Writing
- checks: reachable:✅  no_error_marker:✅  has_answer:✅  banners:✅  sources:—  url_wellformed:—  route:—

Here is a deep dive into the life and legacy of Euclid, carefully grounded in the available historical record and scholarly consensus.

### The Man Behind the Name
Almost nothing is known about Euclid’s personal life with certainty. He is believed to have flourished around 300 BC in Alexandria, Egypt, during the reign of Ptolemy I Soter. The single most important biographical source for Euclid is Proclus (5th century AD), who wrote a *Commentary on the First Book of Euclid's Elements*. Proclus places Euclid after the generation of Plato’s students and before Archimedes, who reportedly references Euclid’s work. His birthplace remains unknown. Medieval Arabic sources sometimes call him "Euclid of Tyre," but this is late and uncertain, and he has often been confused with Euclid of Megara, a Socratic philosopher who lived roughly a century earlier.

Two famous anecdotes are recorded by Proclus and later by Stobaeus: when Ptolemy I asked whether there was an easier path to geometry, Euclid reportedly replied, "There is no royal road to geometry." In another story, a student who had just learned his first theorem asked what profit he would gain from it, and Euclid reportedly told a slave to give him a coin. Neither anecdote can be verified; both may be later inventions meant to illustrate character. Nearly all biographical detail comes from sources writing centuries later, and we do not know where he was born, where he died, who his teachers were, or even whether "Euclid" was a single individual or a name attached to a school of mathematicians working collectively—though the unity of style across his surviving works argues strongly for a single author. Euclid likely worked in Alexandria, possibly associated with the Library and Museum, and Alexandria may have exposed him to a mix of Greek, Babylonian, and Egyptian mathematical traditions, though the exact extent of that influence is uncertain.

### The *Elements*: Structure and Content
Euclid’s monumental work, the *Elements* (Greek: *Stoicheia*), is widely described as one of the most influential textbooks ever written and is often cited as one of the most printed books after the Bible. It is generally compiled around 300 BC. The work consists of 13 books, which survive and are generally considered the complete work as Euclid left it, containing approximately 465 propositions.

Euclid’s genius lay in synthesis and organization rather than original discovery; he gathered the geometrical and number-theoretical knowledge of predecessors including Eudoxus, Theaetetus, Hippocrates of Chios, and others, arranging it into a single, logically deductive edifice. The work proceeds from a small set of definitions, postulates, and "common notions" (axioms) by rigorous proof.

Book I begins with 23 definitions, 5 postulates, and 5 common notions. It culminates in the Pythagorean theorem (I.47) and its converse (I.48). The fifth postulate—the parallel postulate—is more complex and less self-evident than the others: if a straight line falling on two straight lines makes the interior angles on the same side less than two right angles, the two straight lines, if produced indefinitely, meet on that side.

The books cover:
- **Books I–VI:** Plane geometry fundamentals, including triangle congruence, parallelism, area, and circle properties.
- **Books VII–X:** Number theory, covering primes, divisibility, the Euclidean algorithm, and irrational quantities. Book X is the longest and most complex, classifying incommensurable magnitudes and attributed largely to Theaetetus. Some of Euclid’s proofs have been found to contain gaps or implicit assumptions by modern standards.
- **Books XI–XIII:** Solid geometry basics, culminating in the construction of the five Platonic solids inscribed in a sphere and proving that only five exist. The final proposition serves as a majestic conclusion linking geometry to the cosmology of Plato's *Timaeus*.

### Other Works and Historical Transmission
Beyond the *Elements*, Euclid’s other surviving works include *Data*, *On Divisions of Figures* (which survives only in Arabic translation), *Optics*, and *Phenomena*. *Catoptrics* is traditionally attributed to Euclid, but its authorship is disputed; many scholars consider it spurious or by a later writer. Lost works attributed to him include *Conics*, *Porisms*, *Pseudaria* (Book of Fallacies), and *Surface Loci*.

The original Greek text from Euclid’s own hand is lost. The standard Greek text comes mainly through Theon of Alexandria (4th century CE), who edited/revised it; his edition became the ancestor of most surviving manuscripts until 19th-century critical work. Other important Greek manuscripts include the 10th-century Vaticanus Graecus 190.

The *Elements* was translated into Arabic multiple times beginning in the 9th century under the Abbasid Caliphate. Important Arabic translators include al-Hajjaj ibn Yusuf ibn Matar and Ishaq ibn Hunayn, whose version was later revised by Thabit ibn Qurra. In the early 12th century, Adelard of Bath produced a Latin translation from Arabic; other translators, such as Gerard of Cremona, followed. Campanus of Novara produced another influential Latin version around 1260.

The first printed edition (*editio princeps*) was published in 1482 in Venice by Erhard Ratdolt, using Campanus's Latin translation. It was a landmark of early printing, notable for its woodcut diagrams and elegant typography. The first printed Greek text followed in 1533 in Basel by Simon Grynaeus. The definitive Greek text was established by J.L. Heiberg in his edition of 1883–1888. Sir Thomas Heath produced the classic English translation with extensive commentary in 1908 (revised 1926), which remains widely used today.

### Legacy and Influence
The *Elements* was the primary textbook for geometry for over two millennia and was widely used in mathematical education across many cultures. Abraham Lincoln is said to have studied it to sharpen his logical faculties, and Bertrand Russell recalled reading it as a boy and being "dazzled." Euclid’s greatest legacy is the axiomatic-deductive method itself: the idea that a vast body of knowledge can be derived from a small set of starting points by strict logical deduction. This model became the gold standard for rigorous thought. Spinoza reportedly wrote his *Ethics* (1677) *more geometrico*, with definitions, axioms, and propositions. Isaac Newton is also described as structuring his *Principia Mathematica* (1687) on the Euclidean model.

For over 2,000 years, mathematicians suspected the parallel postulate could be proved from the first four. Attempts to prove it were made by Proclus, Omar Khayyam, Nasir al-Din al-Tusi, Girolamo Saccheri, Johann Heinrich Lambert, and Adrien-Marie Legendre, among others. In the 19th century, Carl Friedrich Gauss, János Bolyai, and Nikolai Lobachevsky independently realized that consistent geometries could be constructed by denying it. Lobachevsky published his work on hyperbolic geometry in 1829–1830, and Bolyai's work appeared as an appendix to his father's book in 1832. In hyperbolic geometry, through a point not on a given line there are infinitely many lines parallel to the given line, and the sum of angles in a triangle is less than 180°. Bernhard Riemann later developed elliptic geometry, in which there are no parallel lines and the sum of angles in a triangle exceeds 180°. His 1854 habilitation lecture generalized these ideas into what became Riemannian geometry, which underpins Einstein's general theory of relativity.

Several specific results remain central to mathematics: Euclid’s proof of the infinitude of primes (Book IX, Proposition 20) is one of the most celebrated proofs in mathematics. The Euclidean algorithm (Book VII, Propositions 1–2) is one of the oldest algorithms still in common use, fundamental to number theory and cryptography. The Pythagorean theorem received its most famous classical proof in the *Elements*.

Beyond mathematics, the *Elements* shaped Western epistemology. Kant’s concept of synthetic a priori judgments was partly a response to the status of Euclidean geometry as both necessary and informative about physical space, though this view was later challenged by the development of non-Euclidean geometry. By the late 19th century, the *Elements* began to be displaced as a teaching text. Hilbert’s *Foundations of Geometry* (1899) provided a fully rigorous axiomatization that addressed logical gaps in Euclid. The 'New Math' movement of the mid-20th century further diminished its role in classrooms. Today, it is studied more as a historical monument than as a living textbook, though many high-school geometry courses still reflect Euclidean content and methods. It was part of the curriculum at Harvard and other universities well into the 19th century.

Euclid’s portrait has largely been shaped by later tradition rather than contemporary record, but the framework he systematized fundamentally altered how humanity approaches logic, proof, and the structure of knowledge itself. Let me know if you'd like to explore any specific book, theorem, or historical transmission detail in greater depth.

## Sources
- [Mathematical Association of America, Mathematical Treasure: Euclid's Elementa Geometriae Printed by Ratdolt](https://old.maa.org/press/periodicals/convergence/mathematical-treasure-euclid-s-elementa-geometriae-printed-by-ratdolt)
- [History of Information, Editio Princeps of Euclid's Elements](https://www.historyofinformation.com/detail.php?id=327)
- [Stanford Encyclopedia of Philosophy, Newton's Philosophy (Fall 2024 Edition)](https://plato.stanford.edu/archives/fall2024/entries/newton-philosophy/)
- [Wikipedia: Euclid](https://en.wikipedia.org/wiki/Euclid)
- [Wikipedia: Euclid's Elements](https://en.wikipedia.org/wiki/Euclid%27s_Elements)
- [MathWorld (Wolfram): Euclid](http://mathworld.wolfram.com/Euclid.html)
- [Stedall, Catherine. A Beautiful View of the Elements. — Oxford Reference](https://www.oxfordreference.com/view/10.1093/acref/9780199230138.001.0001/acref-9780199230138-e-579)
- [Purdue University: Euclid's Elements Book I (PDF)](https://www.math.purdue.edu/~goldberg/Math460/Euclid-BKI.pdf)


---
> _Tools used:_
> - **deepseek-v4-pro:cloud** — `web_search` ✅13
> - **glm-5.2:cloud** — `web_search` ✅9
> - **qwen3.6:35b** — `web_search` ✅6, `kb_search` ✅1

---

## bio-pythagoras

- model: `audrey_research`
- status: PASS
- route: research
- latency: route:research  ttft:13.8s  total:439.0s
- banners: Planning → Researching → Verifying → Fact-checking → Writing
- checks: reachable:✅  no_error_marker:✅  has_answer:✅  banners:✅  sources:—  url_wellformed:—  route:—

To give you a clear picture of Pythagoras, we first need to acknowledge a fundamental challenge: it is often reported that he wrote nothing himself, and no contemporary biography of him survives, though the exact nature of these early records remains uncertain. The earliest testimonies about him are frequently described as hostile or indifferent, coming from contemporaries like Xenophanes and Heraclitus, while first sustained biographical traditions are commonly dated to nearly four centuries after his death, appearing in works by Plato, Aristotle, Porphyry, Iamblichus, and Plutarch. Walter Burkert’s influential 1972 study is often cited as concluding that almost nothing reliable can be recovered about Pythagoras himself, only about the later Pythagorean community.

**Origins and Migration**
Pythagoras of Samos is frequently described as a real historical figure who founded a philosophical-religious movement known as "the way of Pythagoras" (*hodos Pythagorike*). He reportedly established his school or community in Croton, a Greek colony in southern Italy (Magna Graecia), attracting cosmopolitan followers from across the Greek world. His birth is commonly dated to around 570 BCE on the island of Samos, though the exact date is unconfirmed. Later sources sometimes name his mother as Pythais and describe his father Mnesarchus as a gem-engraver or wealthy merchant, though some traditions claim he was originally from Tyre in Phoenicia, and his ancestry remains disputed. It is often suggested that Pythagoras left Samos to escape the tyranny of Polycrates, who came to power around 535 BCE, with his migration to Croton generally dated to around 530–520 BCE.

**The Community and Its Doctrines**
Pythagoras reportedly established a community at Croton that became politically influential in the region and neighboring cities. Later sources describe a division within the group into two categories of adherents: the *akousmatikoi* (listeners or hearers), who followed oral precepts, and the *mathematikoi* (learners), who engaged more deeply with mathematical and philosophical study, though whether this split dates to his own time is debated. The community practiced dietary restrictions, including a commonly noted prohibition on eating beans, though the exact rules are often debated. 

Several core doctrines are frequently associated with early Pythagoreanism. Metempsychosis (the transmigration of souls) was central to their practice and is attested from Plato onward, though whether Pythagoras himself taught it or inherited it remains uncertain. The group held a mathematical cosmology, often described as the belief that number is the fundamental principle (*arche*) of all things. Aristotle frequently attributes this doctrine to the Pythagoreans collectively. The discovery that musical intervals correspond to simple whole-number ratios (like 2:1 for an octave) is widely regarded as genuinely Pythagorean, and the related idea of the "harmony of the spheres"—that celestial bodies produce inaudible music through their motions—is also recognized as an early Pythagorean doctrine. One early anecdote, often attributed to Xenophanes, reportedly jokes about Pythagoras stopping a man from beating a dog because he recognized a friend's soul in it.

**Political Conflict and Death**
The community’s political involvement eventually sparked backlash. Croton reportedly won a decisive military victory over the rival city of Sybaris around 510 BCE. Following this, tensions escalated, and a democratic faction led by an aristocrat named Cylon—reportedly rejected from Pythagoras’s inner circle—turned against the group. The anti-Pythagorean attack reportedly occurred around 510 BCE, with meeting houses set on fire and many members killed. According to the late biographer Iamblichus, only two survivors, Lysis and Archippus, escaped the attack, though this detail comes from a heavily embellished source. What happened to Pythagoras himself is uncertain: he reportedly either died during the persecution in Croton or escaped to Metapontum and died there early in the 5th century BCE (c. 495–490 BCE), with some sources placing his death around 490 BCE. A second wave of anti-Pythagorean uprisings is sometimes placed around 450 BCE, though dating remains uncertain.

**Mathematics and the "Pythagorean Theorem"**
The geometric theorem that bears his name did not originate with Pythagoras; Babylonian mathematics (such as the Plimpton 322 tablet, c. 1800 BCE) contains Pythagorean triples centuries before him, and it is often suggested that similar knowledge existed in India long before him. Who actually proved the theorem remains unknown, and Diogenes Laertius’s attribution to Pythagoras is frequently considered a later invention. Euclid’s proof (*Elements* I.47) comes long after his death and likely reflects later Pythagorean work rather than Pythagoras personally proving it. The claim that Pythagoras sacrificed an ox after discovering the theorem is often described as invented or satirical, not historical. 

The Pythagoreans reportedly discovered that irrational numbers exist (e.g., √2), which is often thought to have forced a shift in Greek mathematics from arithmetic to geometry. Over time, "Pythagorean" became a later catch-all label for mathematical and mystical traditions whose provenance cannot be precisely traced.

**Legacy and Later Influence**
Despite the fog of legend, Pythagoras’s intellectual footprint is extensive. His ideas are frequently described as profoundly influencing Plato, particularly in the *Timaeus*, *Republic*, and *Phaedo*. The Pythagorean emphasis on mathematical order also shaped later figures like Philolaus of Croton, who reportedly developed a cosmology organized around a central fire, and Archytas of Tarentum, who made noted contributions to mathematics, music theory, and political philosophy. From the 1st century BCE onward, a neo-Pythagorean movement is commonly dated to reviving and systematizing these ideas, often merging them with Platonism. 

This legacy extended far into the scientific revolution. Copernicus reportedly explicitly acknowledged the Pythagorean heritage of his heliocentric theory in *De revolutionibus orbium coelestium* (1543). Johannes Kepler’s *Mysterium Cosmographicum* (1596/1621) is often cited as proposing that planetary distances could be explained by nesting the five Platonic solids, while his *Harmonices Mundi* (1619) reportedly culminated in his Third Law of Planetary Motion. The doctrine of the harmony of the spheres passed through Plato, neo-Pythagoreans, Boethius, and medieval thought into the Renaissance, shaping music theory, theology, and aesthetics for nearly two millennia.

**History vs. Legend**
To answer your request directly: historically, Pythagoras was likely a charismatic teacher from Samos who founded a tightly knit, politically active community in southern Italy that emphasized mathematical harmony, soul transmigration, and disciplined living. Everything beyond that—the golden thigh, bilocation, specific miracles, exact travel logs to Egypt or Babylon, and personal authorship of theorems—belongs to later hagiography and myth-making. The man who matters most is not the individual whose biography we cannot reliably reconstruct, but the movement he inspired, which fundamentally shaped how Western thought approaches mathematics, cosmology, and the rational structure of reality.

## Sources
- [Pythagoras](https://en.wikipedia.org/wiki/Pythagoras)
- [Pythagoras](https://www.britannica.com/biography/Pythagoras)
- [Pythagoras (570 BC – 490 BC)](https://mathshistory.st-andrews.ac.uk/Biographies/Pythagoras/)
- [Cylon of Croton](https://en.wikipedia.org/wiki/Cylon_of_Croton)
- [Pythagoreanism](https://en.wikipedia.org/wiki/Pythagoreanism)
- [Pythagoras](https://www.encyclopedia.com/people/philosophy-and-religion/philosophy-biographies/pythagoras)
- [Greek Philosopher Pythagoras and His Famous Theorem](https://greekreporter.com/2024/12/16/pythagoras-greek-philosopher-theorem/)
- [The Ashtray: Hippasus of Metapontum (Part 3)](https://archive.nytimes.com/opinionator.blogs.nytimes.com/2011/03/08/the-ashtray-hippasus-of-metapontum-part-3/)


---
> _Tools used:_
> - **qwen3.6:35b** — `web_search` ✅8, `kb_search` ✅1
> - **deepseek-v4-pro:cloud** — `web_search` ✅14
> - **glm-5.2:cloud** — `web_search` ✅14, `kb_search` ✅1

---

## bio-archimedes

- model: `audrey_research`
- status: PASS
- route: research
- latency: route:research  ttft:13.5s  total:371.9s
- banners: Planning → Researching → Verifying → Fact-checking → Writing
- checks: reachable:✅  no_error_marker:✅  has_answer:✅  banners:✅  sources:—  url_wellformed:—  route:—

Here is a deep dive into the life and work of Archimedes, structured by the historical records and surviving manuscripts available today.

### Origins and Early Life
Archimedes is often described as one of the greatest mathematicians of antiquity, though this ranking is subjective and reflects later scholarly opinion rather than a title he held himself. Tradition holds that he was born around 287 BC in Syracuse, Sicily, to his father Phidias, an astronomer. While it is widely reported that Archimedes was nobility and a relative or close friend of King Hiero II of Syracuse, the precise nature of their kinship remains disputed.

His education is reconstructed through later tradition; he likely traveled to Alexandria, Egypt—the intellectual capital of the Hellenistic world—where he may have studied with successors of Euclid. There, he formed lasting friendships with scholars like Eratosthenes and Conon of Samos, establishing a "Republic of Science" across the Mediterranean that allowed him to correspond with peers throughout his life.

### Mathematical Achievements
Archimedes is frequently called the "father of mathematics," a later honorific reflecting his immense influence. His surviving treatises reveal a mind that anticipated integral calculus by nearly two millennia. Key works include:

*   **On the Sphere and Cylinder:** He reportedly proved that the volume of a sphere is two-thirds that of its circumscribing cylinder and that its surface area is four times that of a great circle. This result was so significant to him that he requested a sculpture of a sphere inscribed in a cylinder be carved on his tomb.
*   **Measurement of a Circle:** He established rigorous upper and lower bounds for pi ($\pi$), showing it lies between 223/71 and 22/7 by drawing polygons with up to 96 sides. While highly accurate, this is not confirmed to have been the most precise approximation in existence for over a thousand years.
*   **The Method of Mechanical Theorems:** Rediscovered in the Archimedes Palimpsest—a manuscript identified by the philologist Johan Ludvig Heiberg in 1906—this work reveals that Archimedes used mechanical analogies, such as balancing infinitesimal slices on a lever, to discover geometric results before proving them rigorously.
*   **The Sand Reckoner:** In this treatise, he devised a system for expressing extremely large numbers using exponents, breaking from the traditional Greek number system which stopped at 10,000 (myriad).

### Physics and Engineering: The Engineer of Syracuse
Unlike many of his contemporaries who viewed engineering as manual labor, Archimedes treated mechanics as applied mathematics.

*   **Archimedes' Principle:** In his work *On Floating Bodies*, he stated that an object immersed in fluid is buoyed up by a force equal to the weight of the fluid it displaces. The famous "Eureka" story comes from Vitruvius; while Archimedes reportedly used water displacement to test King Hiero's crown, modern physicists question the practical accuracy of this method for detecting small amounts of silver in an ordinary-sized crown.
*   **The Archimedes Screw:** This device for raising water is traditionally attributed to him by Vitruvius, though its exact origins are debated and it may have been an improvement on existing designs.
*   **War Machines:** During the Siege of Syracuse (214–212 BC), Polybius describes Archimedes' defensive machines causing massive disruption to the Roman fleet. These included grappling cranes ("the Claw") that lifted ships by the prow and catapults capable of striking them at a distance.
*   **The "Heat Ray":** A later tradition claims he used polished bronze mirrors to set Roman ships on fire, but this story does not appear in early accounts like Polybius or Plutarch and is widely considered legendary today.

### Death and Legacy
Archimedes died during the sack of Syracuse in 212 BC. While Roman general Marcus Claudius Marcellus reportedly ordered him spared, a soldier killed him while he was absorbed in a geometric diagram drawn in the dust. The famous dying words, "Do not touch my circles," are likely a later literary embellishment; what matters is that he was simply too focused on his work to respond. Cicero later reported finding his tomb marked with the sphere-and-cylinder sculpture.

His work on statics and hydrostatics is often cited as anticipating calculus by nearly two millennia, influencing later giants like Newton and Galileo. The rediscovery of the Palimpsest in the 20th century confirmed that his thought processes were far more sophisticated than previously known, bridging the gap between pure mathematical theory and practical engineering.

## Sources
- [Archimedes - Wikipedia](https://en.wikipedia.org/wiki/Archimedes)
- [Archimedes (287 BC - 212 BC) - Biography - MacTutor History of Mathematics](https://mathshistory.st-andrews.ac.uk/Biographies/Archimedes/)
- [Archimedes' principle - Wikipedia](https://en.wikipedia.org/wiki/Archimedes'_principle)
- [On Floating Bodies - Wikipedia](https://en.wikipedia.org/wiki/On_Floating_Bodies)
- [Archimedes' On Floating Bodies: The Physics of Buoyancy - Kronecker Wallis](https://www.kroneckerwallis.com/archimedes-on-floating-bodies-the-physics-of-buoyancy/)
- [Archimedes and the Siege of Syracuse: Math Meets Warfare - Kronecker Wallis](https://www.kroneckerwallis.com/archimedes-and-the-siege-of-syracuse-math-meets-warfare/)
- [Siege of Syracuse (Plutarch)](https://math.nyu.edu/Archimedes/Siege/Plutarch.html)
- [Siege of Syracuse (Polybius)](https://math.nyu.edu/Archimedes/Siege/Polybius.html)


---
> _Tools used:_
> - **deepseek-v4-pro:cloud** — `web_search` ✅15
> - **glm-5.2:cloud** — `web_search` ✅6

---

## hist-library-alexandria

- model: `audrey_research`
- status: PASS
- route: research
- latency: route:research  ttft:15.5s  total:358.0s
- banners: Planning → Researching → Verifying → Fact-checking → Writing
- checks: reachable:✅  no_error_marker:✅  has_answer:✅  banners:✅  sources:—  url_wellformed:—  route:—

You asked what we actually know about the Library of Alexandria and which popular claims about its founding and destruction are disputed by historians. Based on the research, here is how modern scholarship breaks down the verified history from the enduring myths.

### What We Can Reasonably Establish
The institution you’re thinking of was part of a larger scholarly complex called the Mouseion (or "shrine of the Muses"), which is often described as having been a state-sponsored research institute in Alexandria’s Bruchion quarter, functioning as a center for textual scholarship, editing, translation, and scientific research under Ptolemaic patronage. It is also reported that the Mouseion may have been founded under Ptolemy I Soter and significantly expanded under Ptolemy II Philadelphus starting around 283 BCE, though this timeline rests on later ancient and medieval sources rather than contemporary administrative records.

Within this complex, there appear to have been two major library collections: a main Royal Library in the Bruchion quarter near the harbor, and a secondary "daughter library" housed in the Serapeum temple in another part of the city, which is thought to have been more accessible and may have served a broader public. Famous scholars like Callimachus, Eratosthenes, Euclid, and Aristarchus are associated with this scholarly environment, and it is noted that scholarly activity in Alexandria may have continued into the 4th century CE, implying some form of library or scholarly infrastructure persisted long after the Ptolemaic period.

### The Decline: A Gradual Process, Not a Single Fire
Rather than vanishing overnight, the Library’s collections are generally understood to have declined gradually over several centuries through neglect, reduced funding, and multiple smaller destructive events, rather than perishing in a single catastrophe. By the time of the Arab conquest in 642 CE, any significant collection in Alexandria was likely a shadow of what it had once been.

Several specific episodes are often cited, though their exact impact remains debated:
- **Caesar’s Fire (48 BCE):** During the Alexandrian War, Julius Caesar reportedly set fire to ships in the harbor, and the flames spread to the city, causing damage to the Library of Alexandria or its scrolls. Ancient sources disagree on the scale: Seneca the Younger, quoting Livy, says 40,000 scrolls were destroyed, while Plutarch claims the entire library burned. Other ancient writers do not mention the library at all in their accounts of Caesar's fire. Some scholars suggest Caesar's fire may have destroyed scrolls stored in warehouses near the port rather than the main library itself. Despite this damage, the Library demonstrably continued to function after Caesar; Mark Antony reportedly gave Cleopatra 200,000 scrolls for it years later.
- **Aurelian’s Campaigns (~270s CE):** During the 3rd century, commonly associated with Emperor Aurelian's campaigns, the Bruchion quarter was devastated. Whether significant collections survived elsewhere at that point remains debated.
- **The Serapeum’s Destruction (391 CE):** Under Emperor Theodosius I, pagan temples were ordered closed or destroyed, and in Alexandria, the Serapeum was reportedly destroyed in ~391 CE by Christians under Patriarch Theophilus. A persistent claim holds that Christians burned the Great Library then, but historians generally challenge this, noting that by 391 the main Royal Library in the Bruchion may have already been long gone or in serious decline, possibly during Aurelian's sack in 273 CE. Whatever scrolls remained in the Serapeum collection were probably lost or dispersed at that point.
- **The Arab Conquest (642 CE):** Alexandria reportedly fell to Arab forces under Amr ibn al-As in 642 CE. Later Arabic sources tell a story that Caliph Umar ibn al-Khattab ordered the library's books burned as fuel for the city's bathhouses. Modern historians overwhelmingly reject this narrative as a late legend; the earliest versions appear centuries after the event in both Muslim and Christian texts, no contemporary or near-contemporary account confirms it, and by 642 CE whatever remained of the ancient library collections was likely very small. Some scholars suggest later chroniclers like Bar Hebraeus may have embellished or transmitted a pre-existing legend to serve polemical purposes, rather than fabricating it outright.

### Popular Claims That Historians Dispute
Several widely circulated stories about the Library do not hold up under historical scrutiny:
- **The Founding Story:** Tradition attributes the Library's founding to Demetrius of Phalerum under Ptolemy I, but this is heavily debated. Demetrius fell out of favor and was exiled by Ptolemy II, making his role in a long-term project unlikely. Many scholars now believe the institution was actually founded or substantially developed under Ptolemy II instead. The story of Demetrius organizing it at Ptolemy I's request is widely treated as a literary embellishment found in later pseudepigraphical texts like the *Letter of Aristeas*, which is also dismissed by scholars as Jewish propaganda written c. 2nd century BCE rather than reliable history.
- **The "Last Librarian" Myth:** Hypatia, the Neoplatonist philosopher reportedly murdered by a Christian mob in 415 CE amid factional/political conflict in Alexandria, is often romantically linked to the Library as its final head librarian. However, there is no ancient source that places her in that role, and historians reject the causal link between her death and the suppression of science or the destruction of the library.
- **The Collection Size:** Ancient sources give wildly inconsistent collection size figures ranging from 40,000 to 700,000 scrolls, but these numbers are widely considered implausible exaggerations by modern scholars like Roger Bagnall, who note that papyrus scroll counts do not map directly onto 'books' in the modern sense and likely included duplicates. A collection in the tens of thousands is considered more realistic.
- **The "Dark Age" Narrative:** The popular framing that the Library's destruction uniquely caused the loss of ancient knowledge and triggered a 'Dark Age' is not supported by historians; most classical texts were lost gradually and through multiple mechanisms across the Mediterranean, not in one Alexandrian catastrophe.

In short, while the Library of Alexandria was undoubtedly a real and magnificent center of Hellenistic scholarship, its history is far more complex than the dramatic myths suggest. The exact fate of its core collection remains genuinely uncertain, with each proposed destruction event having scholars both supporting and contesting its scope. If you'd like to dive deeper into any specific era or scholarly debate surrounding it, just let me know.

## Sources
- [Library of Alexandria (esp. the section on the fate of the library)](https://www.britannica.com/topic/Library-of-Alexandria/The-fate-of-the-Library-of-Alexandria)
- [Library of Alexandria](https://www.worldhistory.org/Library_of_Alexandria/)
- [Alexandria: Library of Dreams (2002)](https://archive.nyu.edu/handle/2451/28263)
- [The Great Myths 5: The Destruction of the Great Library of Alexandria](https://historyforatheists.com/2017/07/the-destruction-of-the-library-of-alexandria/)
- [Evidence from Three Early Historians that the Library of Alexandria was Burned by Amr ibn al-As — History of Information](https://www.historyofinformation.com/detail.php?id=5023)
- [The Myth Of Umar Ibn Al-Khattab: Burning The Library Of Alexandria — Discover the Truth](https://discover-the-truth.com/2017/08/04/the-myth-of-umar-ibn-al-khattab-burning-the-library-of-alexandria/)
- [The Perniciously Persistent Myths of Hypatia and the Great Library — First Things](https://firstthings.com/the-perniciously-persistent-myths-of-hypatia-and-the-great-library/)
- [The Great Myths 9: Hypatia of Alexandria — History for Atheists](https://historyforatheists.com/2020/07/the-great-myths-9-hypatia-of-alexandria/)


---
> _Tools used:_
> - **deepseek-v4-pro:cloud** — `web_search` ✅11
> - **qwen3.6:35b** — `web_search` ✅5
> - **glm-5.2:cloud** — `web_search` ✅10

---

## hist-parallel-postulate

- model: `audrey_research`
- status: PASS
- route: research
- latency: route:research  ttft:0.4s  total:370.9s
- banners: Planning → Researching → Verifying → Fact-checking → Writing
- checks: reachable:✅  no_error_marker:✅  has_answer:✅  banners:✅  sources:—  url_wellformed:—  route:—

You asked about the history of attempts to prove Euclid’s parallel postulate and how that quest unexpectedly birthed non-Euclidean geometry. It is commonly understood that Euclid’s *Elements* (c. 300 BCE) rests on five postulates, with the fifth—the parallel postulate—often described as markedly less self-evident than the others. For roughly two millennia, mathematicians believed it could be derived from the first four, launching a quest that ultimately revealed the postulate’s independence and opened the door to entirely new geometries.

**Ancient & Medieval Attempts**
It is generally reported that Posidonius (c. 135–51 BCE) and Geminus (c. 10 BCE–60 CE) were among the earliest to try proving it. Claudius Ptolemy (c. 90–168 CE) offered a proof that, as later critics showed, unwittingly assumed Playfair’s axiom—a statement equivalent to the postulate itself. It is widely cited that Proclus (412–485 CE) catalogued earlier attempts and gave a false proof; he famously argued that the fifth postulate ought to be struck from the postulates altogether because it should be derivable as a theorem.

During the Islamic Golden Age, Ibn al-Haytham (c. 965–1040) reportedly authored a treatise on the parallel postulate discussing analogous strategies. Omar Khayyam (1048–1131) is said to have analyzed all three cases of what is now called the Saccheri quadrilateral but rejected non-Euclidean conclusions by appealing to his own equivalent postulate about convergent lines. Nasir al-Din al-Tusi (1201–1274) reportedly attempted a proof by contradiction, and his son Sadr al-Din ('Pseudo-Tusi') reportedly wrote a book on the subject in the late 13th century (c. 1298), presenting yet another hypothesis equivalent to the parallel postulate.

**Early Modern Europe**
In the 17th century, John Wallis (c. 1663) reportedly proposed an equivalent form of the postulate—that for any figure, there exists a similar figure of arbitrary size—though this was merely an equivalent restatement rather than a proof from the other axioms. Giordano Vitale (1633–1711) reportedly proved congruence properties of a quadrilateral similar to the later Saccheri quadrilateral in his 1680/1686 works without advancing to non-Euclidean geometry. Leonhard Euler (1707–1783) also proposed multiple proofs that each contained subtle logical gaps only recognized later.

**The Crucial Turning Point**
Giovanni Girolamo Saccheri (1667–1733) is often cited as having published *Euclides ab omni naevo vindicatus* in 1733. He systematically analyzed the quadrilateral’s cases and erroneously claimed to have found a contradiction for the acute-angle case, though he had actually derived theorems that belong to hyperbolic geometry. Around 1766, Johann Heinrich Lambert reportedly took up the same approach. In his unpublished *Theorie der Parallellinien* (published posthumously in 1786), Lambert analyzed the three hypotheses for the angle sum of a triangle but withheld confident publication. Reportedly, he observed that the area of a triangle under the acute-angle hypothesis would be proportional to its angular defect—a profound insight pointing toward hyperbolic geometry. Adrien-Marie Legendre spent roughly forty years wrestling with the problem; his *Éléments de géométrie* first edition is commonly dated to 1794, with multiple subsequent editions over decades. It is generally understood that Legendre proved the angle sum of a triangle cannot exceed 180° under the first four postulates, and that if one triangle has angle sum exactly 180°, then all do—but he could not establish the latter without the postulate.

**The Breakthrough**
By the late 1810s–1820s, it is widely reported that Carl Friedrich Gauss privately became convinced that the parallel postulate was independent of the others and that a consistent non-Euclidean geometry was possible. His private correspondence reportedly includes an 1824 letter to Franz Adolph Taurinus and an 1829 letter to Friedrich Wilhelm Bessel, showing he had worked out the fundamentals of anti-Euclidean or astral geometry. In a letter to Franz Taurinus in 1824, Gauss described a consistent geometry in which the angle sum of a triangle is less than 180°, but asked Taurinus to keep it private, expressing reluctance to publish. Ferdinand Karl Schweikart reportedly developed a similar astral geometry by about 1818, and his nephew Franz Adolph Taurinus reportedly extended this work around 1824–1825, though he reportedly found the implications disturbing and burned some of his own manuscripts. Gauss reportedly never published his findings on non-Euclidean geometry during his lifetime—they became known through correspondence and unpublished notes later edited by others.

The first published account of non-Euclidean geometry is commonly attributed to Nikolai Ivanovich Lobachevsky (1792–1856). It is generally reported that he presented his ideas in 1826 to the university’s faculty and published them in 1829 in the paper *“On the Principles of Geometry”* (*O nachalakh geometrii*), which appeared in the Kazan Messenger around 1829–1830. He later expanded these ideas, publishing *Geometrische Untersuchungen zur Theorie der Parallellinien* in German in 1840 and *Pangeometria* in 1855. Independently, János Bolyai (1802–1860) arrived at the same discovery. It is commonly reported that his work appeared in 1832 as an Appendix appended to his father Farkas Bolyai’s book *Tentamen* (the Appendix dated 1831, printed with the 1832 volume), and he had reportedly been working on non-Euclidean geometry since the mid-1820s. When Farkas sent the Appendix to Gauss in 1832, Gauss reportedly replied that he could not praise it without praising himself, as he had arrived at the same results years earlier. This response is often described as having devastated János Bolyai.

**Consolidation & Riemannian Geometry**
Bernhard Riemann (1826–1866) broadened the revolution. It is generally reported that in his 1854 habilitation lecture *“On the Hypotheses Which Lie at the Foundations of Geometry”* (published posthumously in 1868), he introduced the general concept of an n-dimensional manifold with a metric and classified geometries by curvature. 

A lingering question was whether these new geometries were truly consistent or harbored hidden contradictions. It is commonly understood that Eugenio Beltrami (1835–1900) provided the first proof of relative consistency in 1868. In his *Saggio di interpretazione della geometria Non-Euclidea* and related works, he reportedly showed that hyperbolic geometry is realized on a surface of constant negative curvature—the pseudosphere—and mapped it onto the interior of a unit disk (the Beltrami–Klein model). It is generally reported that this demonstrated non-Euclidean geometry was at least as consistent as Euclidean geometry, marking a decisive step in establishing its logical legitimacy and proving the independence of the parallel postulate. Felix Klein (1849–1925) reportedly further developed the projective model in 1871 and, in his 1872 Erlangen Program, classified geometries by their transformation groups, placing Euclidean and non-Euclidean geometries on equal footing. Building on Beltrami’s earlier rigorous models, Jules Henri Poincaré (1854–1912) introduced the conformal disk model of hyperbolic geometry in 1882, in which hyperbolic straight lines are represented as circular arcs perpendicular to the boundary—a tool that became standard for visualization.

**Why It Matters**
The two-thousand-year quest to prove the parallel postulate thus ended not with a proof, but by showing it is independent of Euclid’s other axioms. This realization opened the door to non-Euclidean geometries that underpin modern physics, from general relativity to cosmology. If you’d like to dive deeper into any specific mathematician’s methods or the geometric models themselves, just let me know.

## Sources
- [Parallel postulate – Wikipedia](https://en.wikipedia.org/wiki/Parallel_postulate)
- [Non-Euclidean geometry – MacTutor History of Mathematics](https://mathshistory.st-andrews.ac.uk/HistTopics/Non-Euclidean_geometry/)
- [Attempts to Prove Euclid's Fifth Postulate](https://www.cut-the-knot.org/triangle/pythpar/Attempts.shtml)
- [Proclus on the Parallel Postulate – MacTutor](https://mathshistory.st-andrews.ac.uk/Extras/Proclus_parallel_postulate/)
- [The Parallel Postulate Problem – Kronecker Wallis](https://www.kroneckerwallis.com/the-parallel-postulate-two-thousand-years-of-trying-to-prove-one-axiom/)
- [Eugenio Beltrami – Wikipedia](https://en.wikipedia.org/wiki/Eugenio_Beltrami)
- [How did Beltrami show the consistency of hyperbolic geometry in his 1868 papers? – Mathematics Stack Exchange](https://math.stackexchange.com/questions/665981/how-did-beltrami-show-the-consistency-of-hyperbolic-geometry-in-his-1868-papers)
- [Eugenio Beltrami – Encyclopedia.com](https://www.encyclopedia.com/people/science-and-technology/mathematics-biographies/eugenio-beltrami)


---
> _Tools used:_
> - **deepseek-v4-pro:cloud** — `web_search` ✅11
> - **glm-5.2:cloud** — `web_search` ✅12
> - **qwen3.6:35b** — `web_search` ✅8, `kb_search` ✅1

---

## current-rust-async

- model: `audrey_research`
- status: PASS
- route: research
- latency: route:research  ttft:12.8s  total:287.4s
- banners: Planning → Researching → Verifying → Fact-checking → Writing
- checks: reachable:✅  no_error_marker:✅  has_answer:✅  banners:✅  sources:—  url_wellformed:—  route:—

Here’s a clear overview of where async runtimes stand in Rust today and how the leading options compare for your projects.

### The Current Landscape
Rust deliberately ships `async`/`await` syntax and the `Future` trait, but it reportedly does not yet include a built-in async executor in the standard library as of mid-2026, though the `polling` crate has been proposed for potential upstream inclusion. Instead, the ecosystem relies on external crates, and the landscape has consolidated significantly.

### Tokio: The Dominant Choice
Tokio is often described as the de facto standard and overwhelmingly dominant async runtime in the Rust ecosystem. It reportedly provides a multi-threaded, work-stealing executor, I/O utilities (TCP, UDP, filesystem, timers, signals), and a vast ecosystem of compatible libraries. Most third-party async crates are commonly assumed to target Tokio by default. 

Architecturally, it uses work-stealing across a thread pool, which reportedly balances load automatically but incurs synchronization overhead. It reportedly works on Linux, macOS, Windows, and other Unixes. Some sources speculate that Tokio 2.0 was released in April 2024, adopting a "single-crate" design that consolidated previous `tokio-*` crates into one package controlled by feature flags, though the official project has reportedly maintained a stable 1.x line with incremental feature additions instead. It is reportedly backed by a large community and corporate sponsors, suggesting its long-term viability is secure.

**Tradeoffs:** Tokio’s ecosystem compatibility is often described as decisive; most async web, gRPC, and database libraries target it. Choosing a non-Tokio runtime reportedly means either using compatibility shims or forgoing popular libraries. It is reportedly large, heavily optimized, and complex; it's battle-tested but hard to fully audit. Its work-stealing scheduler reportedly introduces cross-thread coordination overhead that matters in certain high-throughput or latency-sensitive niches, and the full runtime can feel like overkill for applications that don't need work-stealing.

### smol: The Lightweight Alternative
smol has reportedly emerged as the primary lightweight alternative to Tokio, built on `async-io`/`polling`, designed for minimal footprint, single-core execution, and cross-platform support. It is reportedly actively maintained and reportedly created by Stjepan Glavina. The latest documented version of smol is reportedly 2.0.2. It is described as a thin re-export of several smaller, focused crates (`async-io`, `async-executor`, etc.) that can be used independently or together via the `smol` umbrella crate. The entire runtime is reportedly under 1,500 lines, making it auditable. smol is composable — it works with any `Future`, not just its own types — and supports a configurable thread pool. It reportedly uses a simpler multi-threaded executor by default (with single-threaded options), generally less optimized than Tokio's but with lower overhead for simple use cases. Its I/O backend reportedly uses `async-io` (epoll/kqueue), which is portable and relatively lightweight.

**Tradeoffs:** smol's ecosystem is reportedly much smaller; fewer high-level libraries target it directly, and integration with the Tokio ecosystem reportedly requires adapter layers such as `async-compat`. It can reportedly bridge to Tokio libraries via `async-compat`, though this adds a translation layer. It is reportedly maintained by the smol-rs organization, with a smaller but dedicated community. For many, it is reportedly the best choice when you want a small, auditable runtime and are willing to manage compatibility yourself.

### async-std: Current Status
async-std was reportedly launched in 2019 as an async mirror of the standard library's synchronous APIs, offering a familiar ergonomic surface. It reportedly has been officially discontinued as of 2025 and is no longer maintained; it should be considered legacy. Some sources report that it was officially discontinued on March 1, 2025 (version 1.13.1), with a RUSTSEC advisory recommending smol as its successor, though the exact date remains unconfirmed. It should likely be avoided for new projects unless its status is clarified.

### Specialized Runtimes: Glommio & Monoio
A distinct architectural camp has emerged for high-throughput Linux I/O workloads, built around `io_uring` and a thread-per-core model (no work-stealing).

**Glommio** was reportedly created by Datadog. DataDog describes glommio as a thread-per-core async runtime developed by them, built on Linux's io_uring, featuring priority-based per-core queues and supporting only Linux (x86_64/aarch64). It is described as a cooperative thread-per-core runtime built on Linux's io_uring interface; it reportedly pins threads to cores, avoids helper threads entirely, and relies on cooperative scheduling within each thread. Datadog's blog reportedly describes Glommio as designed for applications where context switches are costly and cache locality is paramount. It is reportedly built against stable Rust with an MSRV of 1.70 per its docs and reportedly remains available on crates.io/docs.rs. Its current maintenance status remains unclear.

**Monoio** (by ByteDance) is described as a thread-per-core runtime built on io_uring (with epoll/kqueue fallbacks). The latest crates.io version of monoio is reportedly 0.2.4. It is part of the CloudWeGo ecosystem (e.g., the Monolake framework). Its thread-per-core design reportedly eliminates work-stealing overhead but requires the application to adopt a sharding/pinned-task model. It reportedly works on Linux (io_uring for best performance) with epoll/kqueue fallbacks for other platforms, though it is reportedly Linux-specific (io_uring is a Linux kernel feature) and generally not portable to other platforms. It is reportedly backed by Bytedance and used in production there, suggesting ongoing investment.

**Tradeoffs:** They use thread-per-core with no work-stealing, reportedly eliminating contention and enabling high throughput for I/O-bound workloads, but require careful application design. The thread-per-core model reportedly maximizes cache locality and minimizes synchronization overhead but sacrifices automatic load balancing; if tasks are unevenly distributed, some cores sit idle while others are overloaded, and the application must reportedly handle work distribution explicitly (e.g., via sharding or connection-aware scheduling). They have much smaller ecosystems; you'll likely be writing more I/O code yourself or using purpose-built frameworks.

### Embassy: The Embedded Frontier
Embassy is described as an async runtime designed specifically for embedded (no-std, bare-metal microcontroller) targets. It reportedly provides async device drivers, a task executor that works without an OS, and tooling for platforms like STM32, nRF, RP2040, and ESP32. It has reportedly gained significant traction in the embedded Rust community by bringing async/await ergonomics to a space that previously relied on manual state machines, interrupts, and callback hell. It reportedly requires no heap (tasks are statically allocated), uses cooperative multitasking with priorities, and has tight hardware integration. It is not a general-purpose OS runtime; it reportedly targets microcontrollers and deeply embedded systems.

### Summary of Main Tradeoffs
- **Ecosystem & Compatibility:** Tokio's ecosystem compatibility is often described as decisive; most async web, gRPC, and database libraries target it. Choosing a non-Tokio runtime reportedly means either using compatibility shims or forgoing popular libraries. smol can reportedly bridge to Tokio libraries via `async-compat`, though this adds a translation layer. monoio and glommio have much smaller ecosystems; you'll likely be writing more I/O code yourself or using purpose-built frameworks.
- **Performance Model:** Tokio uses work-stealing across a thread pool, which reportedly balances load automatically but incurs synchronization overhead. monoio and glommio use thread-per-core with no work-stealing, reportedly eliminating contention and enabling high throughput for I/O-bound workloads, but require careful application design. smol is configurable; by default it's reportedly a simpler multi-threaded executor, generally less optimized than Tokio's but with lower overhead for simple use cases.
- **I/O Model & Portability:** Tokio uses epoll/kqueue/IOCP (traditional readiness-based I/O), which is portable but reportedly incurs syscall overhead per operation. monoio and glommio use io_uring on Linux, which can reportedly dramatically reduce syscall overhead for high-throughput networking and disk I/O; this is Linux-only for the fast path, with monoio reportedly falling back to epoll/kqueue on other platforms. smol uses async-io, which is reportedly epoll/kqueue-based, portable, and relatively lightweight.
- **Complexity & Auditability:** smol is described as tiny and auditable; if you want to understand your entire runtime, smol is the clear choice. Tokio is reportedly large, heavily optimized, and complex; it's battle-tested but hard to fully audit. monoio is moderately sized but reportedly requires understanding io_uring semantics and thread-per-core architecture, which is non-trivial.
- **Maintenance & Longevity:** Tokio is reportedly backed by a large community and corporate sponsors; its long-term viability is not in question. smol is reportedly maintained by the smol-rs organization; it has a smaller but dedicated community. monoio is reportedly backed by Bytedance and used in production there (CloudWeGo), suggesting ongoing investment. glommio's current maintenance status remains unclear.

For most projects, Tokio is often described as the pragmatic default for most projects due to ecosystem gravity. smol is reportedly the best choice when you want a small, auditable runtime and are willing to manage compatibility yourself. monoio (and potentially glommio, if still maintained) is reportedly the right pick when you need maximum Linux I/O throughput and can design your application around thread-per-core task pinning. Specialized runtimes (Glommio, Monoio, Embassy) reportedly persist because they serve architectural niches where Tokio's general-purpose design is a real limitation, not just a preference; the choice is increasingly reportedly driven by concrete workload characteristics rather than taste alone.

Let me know if you'd like a deeper dive into any specific runtime's architecture or migration paths.

## Sources
- [The State of Async Rust: Runtimes](https://corrode.dev/blog/async/)
- [The End of async-std: What Rust Developers Need to Know in 2026](https://wrenlearnsrust.com/posts/the-end-of-async-std.html)
- [Async-std deprecation](https://internals.rust-lang.org/t/async-std-deprecation/23395)
- [Changes/Deprecate async-std](https://fedoraproject.org/wiki/Changes/Deprecate_async-std)
- [glommio (DataDog/glommio)](https://github.com/DataDog/glommio)
- [glommio documentation](https://docs.rs/glommio/latest/glommio/)
- [Tokio runtime docs](https://docs.rs/tokio/latest/tokio/runtime/index.html)
- [Tokio spawn_blocking docs](https://docs.rs/tokio/latest/tokio/task/fn.spawn_blocking.html)


---
> _Tools used:_
> - **deepseek-v4-pro:cloud** — `web_search` ✅14
> - **qwen3.6:35b** — `web_search` ✅8
> - **glm-5.2:cloud** — `web_search` ✅11

---

## current-2025-recent

- model: `audrey_research`
- status: PASS
- route: research
- latency: route:research  ttft:17.3s  total:58.9s
- banners: Planning → Researching → Verifying → Writing
- checks: reachable:✅  no_error_marker:✅  has_answer:✅  banners:✅  sources:—  url_wellformed:—  route:—

---
> _Tools used:_
> - **deepseek-v4-pro:cloud** — `web_search` ✅15

---

## tech-transformer-attention

- model: `audrey_research`
- status: FAIL
- route: research
- latency: route:research  ttft:0.3s  total:0.4s
- banners: Planning → Researching → Verifying → Writing
- checks: reachable:✅  no_error_marker:✅  has_answer:❌  banners:✅  sources:—  url_wellformed:—  route:—

(no answer body)

---

## ctrl-birthday-toast

- model: `audrey_research`
- status: FAIL
- route: research
- latency: route:research  ttft:0.3s  total:0.3s
- banners: Planning → Researching → Verifying → Writing
- checks: reachable:✅  no_error_marker:✅  has_answer:❌  banners:✅  sources:—  url_wellformed:—  route:—

(no answer body)

---

## ctrl-explain-recursion

- model: `audrey_research`
- status: FAIL
- route: research
- latency: route:research  ttft:0.3s  total:0.4s
- banners: Planning → Researching → Verifying → Writing
- checks: reachable:✅  no_error_marker:✅  has_answer:❌  banners:✅  sources:—  url_wellformed:—  route:—

(no answer body)
