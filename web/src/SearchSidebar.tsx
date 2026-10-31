import { useEffect, useRef, useState, type KeyboardEvent } from "react";

import type { SearchResult } from "./api";
import { SearchSourceViewer } from "./SearchSourceViewer";
import { SearchWorkspace } from "./SearchWorkspace";

import "./SearchSidebar.css";

interface SearchSidebarProps {
  open: boolean;
  modal: boolean;
  openRequest?: { query: string; serial: number };
  onClose: () => void;
}

export function SearchSidebar({ open, modal, openRequest, onClose }: SearchSidebarProps) {
  const [selected, setSelected] = useState<SearchResult | null>(null);
  const panel = useRef<HTMLElement>(null);
  const resultTrigger = useRef<HTMLElement | null>(null);
  const wasOpen = useRef(false);

  useEffect(() => {
    let current = true;
    // A new search request returns to results. Collapsing also releases the source
    // reader, while the search workspace keeps its query, filters and scroll.
    queueMicrotask(() => {
      if (!current) return;
      setSelected(null);
      if (open && !wasOpen.current) panel.current?.querySelector<HTMLInputElement>("input[type='search']")?.focus();
      wasOpen.current = open;
    });
    return () => { current = false; };
  }, [open, openRequest?.serial]);

  useEffect(() => {
    if (!open || !modal) return;
    let current = true;
    queueMicrotask(() => {
      if (!current || panel.current?.contains(document.activeElement)) return;
      const target = panel.current?.querySelector<HTMLElement>(".search-source-inline button[aria-label='Back to search results']") ?? panel.current?.querySelector<HTMLInputElement>("input[type='search']");
      target?.focus();
    });
    return () => { current = false; };
  }, [open, modal]);

  function backToResults() {
    setSelected(null);
    queueMicrotask(() => {
      if (open && resultTrigger.current?.isConnected) resultTrigger.current.focus();
    });
  }

  function handleKeyDown(event: KeyboardEvent<HTMLElement>) {
    if (event.key === "Escape") {
      event.preventDefault();
      event.stopPropagation();
      if (selected) backToResults();
      else onClose();
      return;
    }
    if (!modal || event.key !== "Tab") return;
    const items = [...(panel.current?.querySelectorAll<HTMLElement>("button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), a[href], summary, [tabindex='0']") ?? [])]
      .filter((item) => !item.closest("[hidden], [inert]") && (!item.closest("details:not([open])") || item.tagName === "SUMMARY"));
    const first = items[0];
    const last = items.at(-1);
    if (!items.length) {
      event.preventDefault();
      panel.current?.focus();
    } else if (event.shiftKey && (document.activeElement === first || !panel.current?.contains(document.activeElement))) {
      event.preventDefault();
      last?.focus();
    } else if (!event.shiftKey && (document.activeElement === last || !panel.current?.contains(document.activeElement))) {
      event.preventDefault();
      first?.focus();
    }
  }

  return (
    <aside
      id="app-search-sidebar"
      className={`app-search-sidebar${modal ? " app-search-sidebar-modal" : ""}`}
      hidden={!open}
      inert={!open}
      ref={panel}
      role={modal ? "dialog" : "complementary"}
      aria-modal={modal && open ? true : undefined}
      aria-labelledby="app-search-sidebar-title"
      tabIndex={-1}
      onKeyDown={handleKeyDown}
    >
      <header className="app-search-sidebar-header">
        <div><h2 id="app-search-sidebar-title">Search</h2><p>Find your saved work. Search stays local.</p></div>
        <button type="button" aria-label="Collapse Search" onClick={onClose}><span aria-hidden="true">×</span></button>
      </header>
      <div className="app-search-sidebar-results" hidden={selected !== null} inert={selected !== null}>
        <SearchWorkspace
          presentation="sidebar"
          visible={open && selected === null}
          openRequest={openRequest}
          onOpenResult={(result) => {
            resultTrigger.current = document.activeElement instanceof HTMLElement ? document.activeElement : null;
            setSelected(result);
          }}
        />
      </div>
      {open && selected ? <SearchSourceViewer presentation="inline" result={selected} onClose={backToResults} /> : null}
    </aside>
  );
}
