import './style.css';
import { BACKEND_LABEL, initBackend } from './backend';
import { loadDataset } from './data';
import { MODEL_DEFAULTS, countParams } from './model';
import { Trainer, type TrainState, type TrainStats } from './train';

const $ = (testId: string) => document.querySelector<HTMLElement>(`[data-testid="${testId}"]`)!;
const startToggle = $('start-toggle') as HTMLButtonElement;

const fmtLoss = (x: number | null) => (x === null ? '—' : x.toFixed(3));
const pad2 = (n: number) => String(n).padStart(2, '0');
const fmtElapsed = (ms: number) => `${pad2(Math.floor(ms / 60_000))}:${pad2(Math.floor(ms / 1000) % 60)}`;

function renderStats(s: TrainStats): void {
  $('step').textContent = String(s.step);
  $('loss-initial').textContent = fmtLoss(s.initialLoss);
  $('loss-current').textContent = fmtLoss(s.currentLoss);
  $('speed').textContent = s.stepsPerSec.toFixed(1);
  $('elapsed').textContent = fmtElapsed(s.elapsedMs);
}

const TOGGLE_TEXT: Record<TrainState, string> = { idle: '开始训练', running: '暂停', paused: '继续', nan: '继续' };

Promise.all([initBackend(location.search), loadDataset()])
  .then(([be, ds]) => {
    const V = ds.vocab.length;
    $('dataset-info').textContent = `数据集：${ds.poems.length} 首五言绝句 · 字表 ${V} 字 · 来自 chinese-poetry（MIT）`;
    $('loss-baseline').textContent = Math.log(V).toFixed(3);
    $('backend').textContent = BACKEND_LABEL[be.backend];
    const notice = $('backend-notice');
    notice.textContent = be.notices.join('\n');
    notice.hidden = be.notices.length === 0;

    const trainer = new Trainer({
      cfg: { ...MODEL_DEFAULTS, vocabSize: V },
      train: ds.train,
      val: ds.val,
      bosId: ds.bosId,
      onStep: renderStats,
      onState: (s) => (startToggle.textContent = TOGGLE_TEXT[s]),
    });
    $('params').textContent = (countParams(trainer.params) / 1e6).toFixed(2) + 'M';

    // 暂停要等当前这一步结束，按钮文字由 onState 在循环真正停下后更新
    startToggle.addEventListener('click', () => (trainer.state === 'running' ? trainer.pause() : trainer.start()));
    startToggle.textContent = TOGGLE_TEXT[trainer.state];
    startToggle.disabled = false;
  })
  .catch((err) => {
    // ponytail: 只给出文案；「重试」按钮在 Task 13 加
    console.error(err);
    $('dataset-info').textContent = '数据集加载失败';
  });
