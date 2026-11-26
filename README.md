# Tangshi GPT (唐诗 GPT)

Train a ~1M-parameter character-level GPT from random weights, right in your browser, and watch it learn to write Tang-dynasty five-character quatrains (五言绝句).

- **Live training**: loss curve (per step, smoothed EMA, validation loss every 50 steps), step count, steps/s and elapsed time.
- **Samples**: a poem is generated at step 0 and every 50 steps, so you can watch gibberish turn into quatrains.
- **Attention**: heatmaps for all 3 layers × 4 heads, refreshed as the model trains.
- **Nothing to install to use it**: no Python, no GPU drivers, no API key. It is a static web page; the model is written by hand on top of TensorFlow.js core ops in a single file, [`src/model.ts`](src/model.ts) (commented in Chinese).

The UI is in Chinese only.

## About speed

About 1500 steps in roughly 3 minutes is the design target, and it assumes an ordinary laptop with **WebGPU or WebGL hardware acceleration enabled** (≥ 8 steps/s with the default settings). Speed depends entirely on your hardware, so the page shows the real steps/s (「步/秒」) while training. Some measured numbers:

| Setup | Speed |
| --- | --- |
| WebGPU on an Intel Arc 130T integrated GPU | ~7 steps/s |
| Software WebGL (SwiftShader, e.g. headless test runs) | ~0.7 steps/s |
| CPU backend | very slow: ~15 s per step |

## Requirements

- Node.js and npm. Vite needs Node.js 20.19+; the unit tests (Vitest) need 22.12+. Developed on Node.js 22.20.
- Chrome or Edge 113+ recommended (WebGPU). Other browsers fall back to WebGL.

The dataset ships with the repository, so running the app needs no network access.

## Install and run

These commands work as-is in both PowerShell and bash:

```sh
git clone https://github.com/zhushen12580/tangshi-gpt.git
cd tangshi-gpt
npm ci
npm run dev
```

Open http://localhost:5173/ (if that port is busy, Vite picks another one and prints the URL).

### Choosing a backend

By default the page tries WebGPU → WebGL → CPU and uses the first one that passes a quick forward/backward self-test. The badge in the header shows the backend in use. To start from a specific backend, add a query parameter:

- `?backend=webgpu`: same as the default
- `?backend=webgl`: skip WebGPU
- `?backend=cpu`: CPU only (very slow)

If the requested backend fails, the page falls back to the next one and says so. Unknown values are treated as automatic.

## Using the page

- **开始训练 / 暂停 / 继续** (Start / Pause / Resume): start or pause the training loop. **重置** (Reset) re-initializes the model; saved checkpoints are kept.
- **温度** (Temperature): sampling temperature, 0.1–1.5, default 0.8.
- **强制五绝格式** (Force quatrain format): when checked, sampling is constrained to 4 lines of 5 characters with the punctuation 「，。，。」. Uncheck it to see whether the model has learned the format by itself; free samples get a 「格式 ✓」 or 「格式 ✗」 badge.
- **立即生成** (Generate now): write a poem right away, even at step 0 before training. While training, it appears after the current step finishes; training is not interrupted.
- **保存 / 载入 / 删除存档** (Save / Load / Delete checkpoint): one checkpoint slot, stored in your browser's IndexedDB. It holds the weights, the Adam optimizer state, the loss history and the samples. After a reload the page offers to load it (「载入」) or dismiss the prompt (「忽略」).
- **注意力** (Attention):
  - Pick one of the preset poems (登鹳雀楼, 春晓, 相思, 鹿柴) or the latest generated poem (最新生成的诗) to see 12 small heatmaps (3 layers × 4 heads) and an enlarged one.
  - Or type your own poem into **自定义** and press **查看** (View). It must be exactly 20 Chinese characters. Whitespace and punctuation are ignored, and every character must be in the model's vocabulary.
  - Click a small heatmap, or Tab to it and press Enter, to enlarge that head. Focus the enlarged heatmap and use the arrow keys to move the highlighted cell; the weight is shown below it, e.g. 「「海」看「黄」：0.31」.
- **Keyboard**: Tab reaches every button, slider and checkbox. Arrow keys change the slider by 0.1, Space toggles the checkbox, Enter activates buttons.

## Tests

```sh
npm test
npm run test:coverage
npx playwright install chromium
npm run test:e2e
```

- `npm test` runs the Vitest unit tests in Node.
- `npm run test:coverage` checks ≥ 80% line coverage of the core modules.
- `npx playwright install chromium` is only needed once.
- `npm run test:e2e` builds the app, serves it at http://localhost:4173/tangshi-gpt/ and runs the Playwright tests in Chromium. Port 4173 must be free. Training runs on software WebGL, so the whole run takes a while: 23 tests took about 20 minutes on the development machine.

## Build and preview

```sh
npm run build
npm run preview
```

Open http://localhost:4173/tangshi-gpt/ (keep the trailing slash). The output in `dist/` is plain static files that work under any sub-path, e.g. GitHub Pages.

## Dataset

`public/data/poems.json` holds 3672 strict five-character quatrains (5 characters + punctuation × 4 lines), converted to Simplified Chinese. They come from [chinese-poetry](https://github.com/chinese-poetry/chinese-poetry) (MIT) at the fixed commit `b8594f81a89752241442f2ce267d6f66f96704ee`. To rebuild it byte-for-byte (needs network access):

```sh
npm run build:data
```

If `raw.githubusercontent.com` is slow or unreachable, point `CHINESE_POETRY_BASE` at a mirror (or at a local clone of chinese-poetry checked out at that commit).

PowerShell:

```powershell
$env:CHINESE_POETRY_BASE='https://cdn.jsdelivr.net/gh/chinese-poetry/chinese-poetry@b8594f81a89752241442f2ce267d6f66f96704ee/'
npm run build:data
```

bash:

```sh
CHINESE_POETRY_BASE='https://cdn.jsdelivr.net/gh/chinese-poetry/chinese-poetry@b8594f81a89752241442f2ce267d6f66f96704ee/' npm run build:data
```

Afterwards `git status --porcelain public/data` should print nothing. Source and license details are in [`public/data/NOTICE.txt`](public/data/NOTICE.txt).

## License

MIT, see [LICENSE](LICENSE). The dataset is from chinese-poetry, also MIT.

Built with Claude Code & Codex
