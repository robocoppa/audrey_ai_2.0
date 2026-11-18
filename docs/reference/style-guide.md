# Audrey native app style guide

This guide covers the browser app in `web/`: Chat, Agents, Watches, Projects,
Search, files, account settings and administration. Keep the current dark navy
surfaces, blue accents, compact controls and quiet hover treatment consistent.
Use the shared system when adding or changing a screen.

## Sources of truth

| File | Responsibility |
| --- | --- |
| [theme.css](../../web/src/theme.css) | Shared colors, typography, spacing, radii, control sizes, shadows, motion and layers |
| [styles.css](../../web/src/styles.css) | Global defaults, app shell and established Chat/account/file layouts |
| [SidebarCreateButton.tsx](../../web/src/SidebarCreateButton.tsx) | Shared sidebar creation action for conversations, Agents and Watches |
| Feature CSS beside each component | Component arrangement, responsive layout and role-specific token usage |

Theme tokens are CSS custom properties on `:root`. Components reference those
properties with `var(...)`. Change a shared value at its source instead of copying
a new color or slightly different control style into each workspace.

The style guide defines shared choices; it does not make every layout dimension
global. A document's aspect ratio, a table's minimum column width and a drawer's
placement still belong to the component that owns that layout.

## Colors and meaning

Use tokens by purpose, rather than choosing a color because it happens to look
similar. Keep brand, status and document colors separate.

| Role | Tokens and starting colors | Use |
| --- | --- | --- |
| Background hierarchy | `--surface-deep` `#07101f`; `--surface` `#0f1c31`; `--surface-raised` `#152641`; `--surface-active` `#183153` | Canvas, panels, elevated surfaces and selected items |
| Fields | `--input-surface` `#101f36` | Inputs where a distinct editable surface helps |
| Text hierarchy | `--text` `#f4f8ff`; `--text-soft` `#d7e2f1`; `--muted` `#9bacc4`; `--muted-deep` `#879ab5` | Main text, secondary content and supporting metadata |
| Brand accent | `--accent` `#63b5ff`; `--accent-soft` `#3499fa` | Links, active controls, focus and blue fills |
| Gradient | `--brand-middle` `#4676f2`; `--brand-end` `#5158ec`; `--brand-gradient` | Gradient beginning at `--accent-soft`, through the middle and end colors |
| On accent | `--on-accent` `#fff`; `--on-accent-dark` `#050f1e` | Foreground on accent surfaces |
| Status | `--success` `#68d391`; `--warning` `#efc071`; `--danger` `#ff91a4` | Success, attention and error/destructive states |
| Private | `--private` `#c3b5ed` | Private-mode distinction, paired with a label |
| Dividers | `--border`, `--border-strong` | Quiet separation and stronger control outlines |
| Scrolling | `--scrollbar-thumb`, `--scrollbar-thumb-hover` | Thin, unobtrusive scroll indicators |

The background hierarchy should remain visible without making each card look
like a separate floating window. Metadata is quieter than task content, but must
remain readable. Pair status color with a label or icon; color alone must not
communicate Running, Failed, Private or deletion confirmation.

For translucent treatments, derive the tint from its semantic color:

```css
.example-selection {
  color: var(--text);
  border-color: color-mix(in srgb, var(--accent) 35%, transparent);
  background: color-mix(in srgb, var(--accent-soft) 12%, transparent);
}
```

Do not duplicate a color as both a hex value and unrelated RGB channel variables.
The tint should follow its base color automatically. Reuse an existing surface,
fill or border token when it already represents the intended role.

### Changing the app palette

1. Edit the semantic colors in `theme.css`.
2. Adjust the brand gradient's stop tokens when changing the creation-button
   gradient. An accent change should not silently turn success or errors blue.
3. Check normal, hover, selected, focus and disabled states on both dark panels
   and raised surfaces.
4. Check text, controls and status labels at phone and laptop widths, including
   Search open and modal/drawer views.

Use a browser's temporary `:root` overrides while exploring a palette, then put
the accepted values in `theme.css`. Avoid per-component theme overrides unless
the component has a distinct semantic surface, such as a document reader.

### Documents and external content

Document-paper and document-ink colors are separate from the dark app chrome.
Markdown/Office previews and spreadsheet grids need readable light-page and
table treatments; use the `--document-*` and `--sheet-*` families rather than app text or brand
tokens. System colors used by forced-colors overrides remain system colors.

Theme changes do not recolor uploaded images, PDF page pixels, video frames,
provider marks or colors supplied by a document. Do not apply a blanket filter
to a preview in order to make it match the app palette.

## Typography, spacing and shape

Use `--font-sans` for interface text, `--font-mono` for code and fixed-width
values, and `--font-document` for the document reader. Choose
the role first: metadata, control label, body content or heading. Keep source
text, code and document-relative `em` sizes readable within their own context.

| Size token | Starting size | Typical use |
| --- | --- | --- |
| `--font-size-xs` | `0.7rem` | Compact metadata |
| `--font-size-sm` | `0.75rem` | Supporting labels |
| `--font-size-label` | `0.78rem` | Creation/action labels |
| `--font-size-body-sm` | `0.8rem` | Compact body content |
| `--font-size-body` | `0.85rem` | Interface body text |
| `--font-size-base` | `1rem` | Base text and larger controls |
| `--font-size-title` | `1.3rem` | Section/form titles |

Weights are `--font-weight-medium` (500), `--font-weight-semibold` (600) and
`--font-weight-bold` (700). Use `--line-height-ui` (1.5) for controls and
`--line-height-reading` (1.65) for longer content.

Prefer shared spacing tokens for repeated padding, gaps and vertical rhythm.
Related controls sit closer together than separate sections. Keep the app
compact enough for a laptop without removing readable line height or keyboard
focus space. Use layout-specific values for geometry, not as an alternate global
spacing scale.

The spacing scale is `--space-1` through `--space-6` at 0.25rem increments,
then `--space-8` (2rem), `--space-10` (2.5rem) and `--space-12` (3rem).

Use radius tokens for controls, cards, dialogs and pills. Use the control-height
tokens for consistent action sizes; tiny icon targets should not become difficult
to activate. Shadows indicate elevation or a restrained hover glow. Avoid adding
new shadow recipes to make an ordinary panel look more prominent.

| Shape/size role | Tokens and starting values |
| --- | --- |
| Radius | `--radius-sm` 0.5rem; `--radius-control` 0.65rem; `--radius-button` 0.75rem; `--radius-panel` 1rem; `--radius-pill` 999px |
| Control height | `--control-height-sm` 2.2rem; `--control-height-md` 2.5rem; `--control-height-lg` 2.75rem |
| Sidebar creation | `--control-height-create` 2.65rem; `--control-gap-create` 0.45rem; `--control-padding-create` 0.65rem 0.85rem |
| Navigation width | `--workspace-sidebar-width` 17rem |
| Elevation | `--shadow-sm`, `--shadow-md`, `--shadow-create-hover` |

## Buttons and controls

| Role | Treatment |
| --- | --- |
| Sidebar creation | Shared `SidebarCreateButton`: plus icon, brand gradient, white foreground and the established New conversation dimensions |
| Main action | Filled accent treatment for the action that moves the current form or workflow forward |
| Secondary action | Quieter fill/border for supporting actions such as Back, Edit and Cancel |
| Header/navigation | Compact surface-based control; hover fills gently and adds a slight glow |
| Destructive action | Danger text/border/fill; preserve the existing two-click checkmark confirmation flow where present |
| Text action | Quiet text treatment for Refresh, disclosure links and low-emphasis navigation |

Use the shared creation component for New conversation, New agent and New watch.
Pass the action, disabled state and accessible name through normal button props.
Do not layer another primary-button paint class over it: one component owns the
normal, hover, focus and disabled appearance.

Hover should feel like a gentle filled glow, following My Files and the existing
header buttons. Avoid a bright rim around an otherwise unchanged control.
Creation buttons keep the established slight brightness increase and soft shadow.

Keyboard focus is a separate state: use the shared focus outline and offset so
it remains visible beside hover and selection. Do not remove a focus outline
merely to make the hover state quieter. Disabled controls keep their label and
shape, use the shared opacity and do not react as enabled controls on hover.

Inputs, selects and textareas use shared surface, text, border, radius and focus
tokens. Keep validation text beside its field or inside the matching card.
Preserve drafts on conflicts; styling must not change submission, cancellation
or deletion behavior.

## Panels, navigation and responsive layout

Use the shared sidebar-width token where Chat, Projects and Search share that
navigation width. Keep headings fixed when a panel owns a scrolling body.
Independent sidebar, workflow and activity scrolling should stay independent.
Use `min-width: 0` and `min-height: 0` where flex/grid children need to shrink;
do not mask broken layout with a global horizontal-overflow rule.

Responsive changes follow the space a component actually needs. Existing key
boundaries include 760px for Agent/Watch controls drawers, 1100px for compact
header controls, and the 1280px Search docking decision. Document/file controls
have smaller local breakpoints. Preserve those contracts when adjusting tokens.

Ordinary CSS custom properties cannot replace values inside media-query
conditions. Keep those conditions explicit, and update matching `matchMedia` or
responsive React logic together when changing a shared boundary.

Use shared layer tokens for overlays, navigation drawers and dialogs. A new
overlay should sit in the intended layer; avoid solving a local stacking-context
problem by inventing a larger global `z-index`.

The layer order is `--z-popover` (30), `--z-search` (40),
`--z-drawer-backdrop` (60), `--z-drawer` (61), `--z-dialog` (70),
`--z-file-preview` (75), `--z-source-preview` (80) and
`--z-knowledge-dialog` (90). Local sticky rows may still use small local layers.

## Accessibility and motion

- Keep visible labels or accessible names on icon actions. Decorative plus and
  chevron icons stay hidden from assistive technology.
- Use semantic buttons and native disabled states. Navigation selection, expanded
  state and destructive confirmation must remain available without relying on color.
- Keep focus out of hidden or inert workspaces and return it to a useful trigger
  after closing a drawer or dialog.
- Use shared motion durations/easing, and respect `prefers-reduced-motion` for
  transitions and decorative animation. Motion must not be needed to understand state.
- Preserve forced-colors overrides with system colors such as CanvasText and
  Highlight. Test focus, borders and document previews in that mode.

Motion uses `--duration-fast` (120ms), `--duration-normal` (180ms),
`--duration-slow` (240ms) and `--ease-standard`. Reduced motion sets these
durations to zero; decorative keyframe animations also need their own reduced
motion treatment. Focus uses `--focus-ring` and `--focus-offset`.

## Adding or changing styles

Choose an existing semantic token and shared component before adding a new one.
If a new role is needed, name it after its purpose and define it in `theme.css`.
Avoid names tied to one hex value, one screen or a guessed shade number. Feature
CSS should explain arrangement and state, rather than becoming another palette.

Keep visual changes separate from authorization, schedule execution and data
lifecycle behavior. A new appearance must not alter a confirmation sequence,
start an Agent, activate a Watch or discard a draft implicitly.

Verify the changed component's real states in the browser: normal, hover,
keyboard focus, disabled, selected, empty, loading and error where relevant.
Check phone and laptop widths, long labels/content and a temporary palette change
to confirm dependent fills and borders follow their base tokens. Run the relevant
frontend checks and production build; scripted fixtures supplement the owner's
real interaction check.

From `web/`, check that CSS colors remain centralized and variable references
resolve:

```bash
npm run check:styles
```

The production build runs this guard automatically before TypeScript and Vite.
It checks literal hex/functional CSS colors and undefined variable references;
it does not validate every CSS syntax or named-color choice.

Then run type checking, changed-file lint and the affected component tests using
the scripts in [package.json](../../web/package.json). Before shipping CSS or
component changes, verify the production bundle:

```bash
npm run build
```

The style check is a drift guard, not a contrast or layout test. Browser checks
still establish whether a recolor, hover state, drawer or document is usable.
