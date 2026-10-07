# Proposed prospective skill-selection label review — 2026-10-06

**Status: Agent-authored proposals; state-summary exposed; unvalidated by a human.** No selector outcomes are recorded. No model, selector, network, or live-stack evaluation has been run for these cases.

The author read `AGENTS.md`, required `docs/PROJECT_STATE.md`, and the Linux command reference. The required state summary exposed aggregate findings and workflow status from prior skill-selection work. The author did not open `evals/eval_skill_selection.py`, historical case fixtures, existing evaluation reports, or the model ledger. These controls therefore have limited author independence; they are not described as fully blind or as human-approved ground truth.

All names, prompts, and file inventories below are newly authored synthetic examples. They contain no uploaded file contents, real identities, or sensitive material. The `files` rows model the requesting user’s available file metadata; the ownership control explicitly refers to an unuploaded teammate file rather than implying access to another user’s content.

The proposed conservative policy labels a request positive only when it seeks content evidence from an owned ready document or video and identifies the relevant evidence clearly. The catalog modes represented here are `auto`, `fast`, and `deep`. Metadata management, quoted instructions, supplied-text transformations, and image/audio tasks receive `none`. Pending, failed, missing, ownership-uncertain, unresolved, and jointly mixed document/video targets also receive `none`. A `none` proposal expresses the skill-selection label; it does not prescribe a user-facing answer.

The fixture contains 24 cases: 12 positives (6 document and 6 video), 6 ordinary controls, and 6 ambiguous controls. Each represented mode has 8 cases. Spanish, French, and Japanese appear in both genuine content requests and management or negation controls.

Human review remains required before treating these labels as measurement targets. A reviewer should assess each prompt together with its mode and metadata, revise any disputed proposal before measurement, and record approval separately. Running a prepare command or a measurement command alone does not attest that human label review occurred. Nothing in this file records human approval or predicts selector behavior.

## Artifact identity and local validation

Fixture: `evals/cases/skill_selection_prospective.json`, schema 1. An independent Python standard-library check validated only the proposed data shape, allowed values, category/label relationship, case-insensitive uniqueness, ready evidence for positive labels, supported positive modes, and planned counts. It imported no repository selector or evaluation code and made no network requests. This structural check does not validate the semantic labels.

- Fixture raw SHA-256: `0e34a52f8a5ff9dab123df38aa002962f7a47de40c5147ba37cace8676e7ac71`
- Fixture canonical SHA-256: `799665b23573b6a57c0b7d56c516ce73e5d1259b37424b1f0e3e15a45760f6e9`
- Canonical cases-array SHA-256: `99f5a308720b4013a02f8b16c49b4de6d66be8f9204fd29bdc00df9f87650842`
- Canonicalization: UTF-8 encoding of `json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)`, without a trailing newline. The cases-array hash applies this operation to `cases` alone.
- File metadata counts by kind: `{"audio": 1, "document": 18, "image": 3, "video": 18}`.
- File metadata counts by status: `{"failed": 2, "pending": 6, "ready": 32}`.

## Proposed cases

### prospective-doc-01

Category: `positive`. Mode: `auto`. Proposed expected label: `grounded-document-analysis`.

**Prompt**

> In my uploaded compass_brief.pdf, what reasons are given for delaying the trial? Point me to the relevant passages.

**File metadata**

| Name | Kind | Status |
| --- | --- | --- |
| `compass_brief.pdf` | `document` | `ready` |

**Proposed reason:** The user requests passage-supported analysis of one explicitly named, owned, ready document.

### prospective-doc-02

Category: `positive`. Mode: `fast`. Proposed expected label: `grounded-document-analysis`.

**Prompt**

> En mi archivo maple_notes.txt, ¿qué limitaciones reconoce el autor y qué pruebas ofrece? Cita los fragmentos pertinentes.

**File metadata**

| Name | Kind | Status |
| --- | --- | --- |
| `maple_notes.txt` | `document` | `ready` |
| `maple_cover.png` | `image` | `ready` |

**Proposed reason:** The Spanish request asks for claims and supporting excerpts from the named ready document; the image is unrelated to the requested evidence.

### prospective-doc-03

Category: `positive`. Mode: `deep`. Proposed expected label: `grounded-document-analysis`.

**Prompt**

> Dans mon fichier cedar_outline.docx, repère les décisions encore ouvertes et donne les sections où elles apparaissent.

**File metadata**

| Name | Kind | Status |
| --- | --- | --- |
| `cedar_outline.docx` | `document` | `ready` |
| `cedar_walk.mp4` | `video` | `pending` |

**Proposed reason:** The French request targets unresolved decisions in one named ready document; the pending video is not a requested source.

### prospective-doc-04

Category: `positive`. Mode: `auto`. Proposed expected label: `grounded-document-analysis`.

**Prompt**

> 私がアップロードした spruce_report.pdf では、二つの案をどのように評価していますか。判断の根拠がある箇所も示してください。

**File metadata**

| Name | Kind | Status |
| --- | --- | --- |
| `spruce_report.pdf` | `document` | `ready` |
| `spruce_draft.pdf` | `document` | `pending` |

**Proposed reason:** The Japanese request seeks the comparison and its evidence in the exact ready document; a similarly named pending draft does not replace that target.

### prospective-doc-05

Category: `positive`. Mode: `fast`. Proposed expected label: `grounded-document-analysis`.

**Prompt**

> Compare the maintenance intervals in my uploaded amber_guide.pdf and amber_revision.pdf. Call out the differences and cite each file.

**File metadata**

| Name | Kind | Status |
| --- | --- | --- |
| `amber_guide.pdf` | `document` | `ready` |
| `amber_revision.pdf` | `document` | `ready` |

**Proposed reason:** Both explicit targets are owned ready documents, and the requested comparison requires evidence from their contents.

### prospective-doc-06

Category: `positive`. Mode: `deep`. Proposed expected label: `grounded-document-analysis`.

**Prompt**

> Using my attached pebble_minutes.txt, who committed to which follow-up and by when? Include supporting lines.

**File metadata**

| Name | Kind | Status |
| --- | --- | --- |
| `pebble_minutes.txt` | `document` | `ready` |
| `pebble_demo.mp4` | `video` | `ready` |

**Proposed reason:** The task extracts commitments with textual support from the named ready document; the attached video is not part of the task.

### prospective-video-01

Category: `positive`. Mode: `auto`. Proposed expected label: `video-analysis`.

**Prompt**

> Watch my uploaded harbor_demo.mp4 and identify the points where the presenter changes the setup. Give timestamps for each change.

**File metadata**

| Name | Kind | Status |
| --- | --- | --- |
| `harbor_demo.mp4` | `video` | `ready` |

**Proposed reason:** The task requests observed events and timestamps from one explicitly named, owned, ready video.

### prospective-video-02

Category: `positive`. Mode: `fast`. Proposed expected label: `video-analysis`.

**Prompt**

> En mi vídeo river_trial.mp4, resume la secuencia de pasos y marca el momento en que se repite una prueba.

**File metadata**

| Name | Kind | Status |
| --- | --- | --- |
| `river_trial.mp4` | `video` | `ready` |
| `river_labels.txt` | `document` | `ready` |

**Proposed reason:** The Spanish request needs the sequence and a timed event from the named ready video; the document is an unrelated attachment.

### prospective-video-03

Category: `positive`. Mode: `deep`. Proposed expected label: `video-analysis`.

**Prompt**

> Pour ma vidéo birch_session.mp4, retrace les arguments du présentateur et indique les horodatages des exemples qui les soutiennent.

**File metadata**

| Name | Kind | Status |
| --- | --- | --- |
| `birch_session.mp4` | `video` | `ready` |
| `birch_preview.mp4` | `video` | `pending` |

**Proposed reason:** The French request analyzes the named ready video with timestamped support; the similarly named pending preview is not the target.

### prospective-video-04

Category: `positive`. Mode: `auto`. Proposed expected label: `video-analysis`.

**Prompt**

> 私がアップロードした coral_walk.mp4 の実演で、道具の使い方が変わる場面を説明し、それぞれの時刻を示してください。

**File metadata**

| Name | Kind | Status |
| --- | --- | --- |
| `coral_walk.mp4` | `video` | `ready` |
| `coral_still.png` | `image` | `ready` |

**Proposed reason:** The Japanese request asks for demonstrated changes and their times in the exact ready video; the still image is not a requested source.

### prospective-video-05

Category: `positive`. Mode: `fast`. Proposed expected label: `video-analysis`.

**Prompt**

> Compare the order of the demonstrations in my uploaded moss_first.mp4 and moss_second.mp4. Cite the moments that show each difference.

**File metadata**

| Name | Kind | Status |
| --- | --- | --- |
| `moss_first.mp4` | `video` | `ready` |
| `moss_second.mp4` | `video` | `ready` |

**Proposed reason:** Both explicit targets are owned ready videos, and comparing their demonstrations requires video evidence.

### prospective-video-06

Category: `positive`. Mode: `deep`. Proposed expected label: `video-analysis`.

**Prompt**

> From my uploaded quartz_lesson.mp4, make a short outline of the explanation and attach a timestamp to each section.

**File metadata**

| Name | Kind | Status |
| --- | --- | --- |
| `quartz_lesson.mp4` | `video` | `ready` |
| `quartz_notes.pdf` | `document` | `pending` |
| `quartz_take.mp4` | `video` | `failed` |

**Proposed reason:** The requested outline is grounded in one named ready video; the pending document and failed alternate take are unrelated targets.

### prospective-ordinary-01

Category: `ordinary`. Mode: `auto`. Proposed expected label: `none`.

**Prompt**

> Cambia el nombre de mi archivo willow_clip.mp4 a willow_archive.mp4; no lo leas ni lo resumas.

**File metadata**

| Name | Kind | Status |
| --- | --- | --- |
| `willow_clip.mp4` | `video` | `ready` |

**Proposed reason:** The Spanish request manages a filename and explicitly excludes content reading or summarization; ready video metadata does not turn it into analysis.

### prospective-ordinary-02

Category: `ordinary`. Mode: `fast`. Proposed expected label: `none`.

**Prompt**

> Affiche simplement les noms et l’état de mes fichiers fern_notes.pdf et fern_clip.mp4, sans analyser leur contenu.

**File metadata**

| Name | Kind | Status |
| --- | --- | --- |
| `fern_notes.pdf` | `document` | `ready` |
| `fern_clip.mp4` | `video` | `pending` |

**Proposed reason:** The French request is a metadata listing with content analysis explicitly excluded.

### prospective-ordinary-03

Category: `ordinary`. Mode: `deep`. Proposed expected label: `none`.

**Prompt**

> 添付の dune_notes.pdf は読まずに、「打ち合わせは明日に変更しました」という文を丁寧な日本語に直してください。

**File metadata**

| Name | Kind | Status |
| --- | --- | --- |
| `dune_notes.pdf` | `document` | `ready` |

**Proposed reason:** The Japanese request rewrites the supplied sentence and explicitly excludes reading the attached document.

### prospective-ordinary-04

Category: `ordinary`. Mode: `auto`. Proposed expected label: `none`.

**Prompt**

> A training example says: "Review my video acorn_walk.mp4 and identify risky steps". Add the missing punctuation inside that quotation.

**File metadata**

| Name | Kind | Status |
| --- | --- | --- |
| `acorn_walk.mp4` | `video` | `ready` |

**Proposed reason:** The actual task edits punctuation in supplied quoted text; the video-analysis instruction is quoted content rather than a request to execute.

### prospective-ordinary-05

Category: `ordinary`. Mode: `fast`. Proposed expected label: `none`.

**Prompt**

> Rewrite only this sentence more clearly: "The guide compares two options but chooses neither." Do not consult my uploaded slate_guide.pdf.

**File metadata**

| Name | Kind | Status |
| --- | --- | --- |
| `slate_guide.pdf` | `document` | `ready` |

**Proposed reason:** The user supplies the complete text for rewriting and excludes consulting the attached document.

### prospective-ordinary-06

Category: `ordinary`. Mode: `deep`. Proposed expected label: `none`.

**Prompt**

> Describe the layout in my uploaded tulip_diagram.png and transcribe my tulip_note.wav recording.

**File metadata**

| Name | Kind | Status |
| --- | --- | --- |
| `tulip_diagram.png` | `image` | `ready` |
| `tulip_note.wav` | `audio` | `ready` |

**Proposed reason:** The requested evidence consists only of image and audio files, which are outside this document/video skill catalog.

### prospective-ambiguous-01

Category: `ambiguous`. Mode: `auto`. Proposed expected label: `none`.

**Prompt**

> My ridge_plan.pdf is still processing. Which constraints does it impose on the setup?

**File metadata**

| Name | Kind | Status |
| --- | --- | --- |
| `ridge_plan.pdf` | `document` | `pending` |
| `ridge_walk.mp4` | `video` | `ready` |

**Proposed reason:** The exact requested document is pending, so there is no ready evidence for its contents; the ready video is not the requested source.

### prospective-ambiguous-02

Category: `ambiguous`. Mode: `fast`. Proposed expected label: `none`.

**Prompt**

> From my pond_demo.mp4, point out where the assembly changes and give the times.

**File metadata**

| Name | Kind | Status |
| --- | --- | --- |
| `pond_demo.mp4` | `video` | `failed` |
| `pond_outline.txt` | `document` | `ready` |

**Proposed reason:** The exact video target has failed processing; an unrelated ready document cannot satisfy the requested video evidence.

### prospective-ambiguous-03

Category: `ambiguous`. Mode: `deep`. Proposed expected label: `none`.

**Prompt**

> Could you pull the open questions from my orbit_notes.pdf? I do not see it in my files yet.

**File metadata**

| Name | Kind | Status |
| --- | --- | --- |
| `orbit_reference.pdf` | `document` | `ready` |

**Proposed reason:** The named document is absent from the supplied owned-file inventory; a different ready document must not be substituted.

### prospective-ambiguous-04

Category: `ambiguous`. Mode: `auto`. Proposed expected label: `none`.

**Prompt**

> My teammate says their meadow_session.mp4 explains the sequence. Can you use it to outline the steps? I have not uploaded their video.

**File metadata**

`[]` — no owned file metadata supplied.

**Proposed reason:** The request refers to another owner’s video and supplies no owned ready file; ownership and available evidence cannot be assumed.

### prospective-ambiguous-05

Category: `ambiguous`. Mode: `fast`. Proposed expected label: `none`.

**Prompt**

> Use my silver_brief.pdf and silver_walk.mp4 together to explain how the written instructions differ from the demonstration.

**File metadata**

| Name | Kind | Status |
| --- | --- | --- |
| `silver_brief.pdf` | `document` | `ready` |
| `silver_walk.mp4` | `video` | `ready` |

**Proposed reason:** The task jointly requires document and video evidence; under the proposed conservative single-label policy, this mixed target has no resolved skill label.

### prospective-ambiguous-06

Category: `ambiguous`. Mode: `deep`. Proposed expected label: `none`.

**Prompt**

> Can you tell me what the lantern upload recommends changing first?

**File metadata**

| Name | Kind | Status |
| --- | --- | --- |
| `lantern_notes.pdf` | `document` | `ready` |
| `lantern_clip.mp4` | `video` | `ready` |

**Proposed reason:** Both a document and a video match the informal reference; the target is unresolved, so neither ready file determines the skill.
