# 🏰 Module 11: Capstone Project - The Interactive Storybook App

> **Central Anchor in this Module:** Packaging our trained, aligned, and KV-cached `StoryCraft-LLM` into a real-time web application featuring token streaming, character pickers, story themes, and a "Choose-Your-Own-Adventure" interactive branching mode.

---

## 🎯 Learning Objectives
1. Build a high-performance streaming backend using FastAPI and Server-Sent Events (SSE).
2. Create an intuitive, beautiful frontend with real-time typewriter token streaming.
3. Add interactive features:
   - **Story Customizer:** Select protagonist name, animal companion, setting, and moral lesson.
   - **Branching Stories:** Click on any sentence in the story to branch into alternate timelines.
   - **Storybook Export:** Export generated stories with formatted text to PDF/Markdown.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       Client Browser                        │
│   (HTML5 / CSS / Vanilla JS EventSource / Modern UI)        │
└───────────────▲─────────────────────────────┬───────────────┘
                │                             │
    Token Stream│                             │ HTTP POST Request
    (text/event-stream)                       │ {"prompt": "...", "moral": "..."}
                │                             ▼
┌───────────────┴─────────────────────────────────────────────┐
│                     FastAPI Server                          │
│   - Receives story prompt & parameters                      │
│   - Formats ChatML context template                         │
│   - Yields tokens asynchronously via generator              │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 StoryCraft Engine (PyTorch)                 │
│   - StoryCraftLM (Module 03)                                │
│   - Aligned Weights (Module 07)                             │
│   - KV-Cached Streaming Generator (Module 09)               │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Running the Capstone App

### 1. Launch Server
```bash
python server.py --checkpoint ../../tinystory_model/storycraft_aligned.pt --port 8000
```

### 2. Open in Browser
Visit `http://localhost:8000` to interact with your custom-built AI Storyteller!
