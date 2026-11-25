import './style.css';
import { BACKEND_LABEL, initBackend } from './backend';
import {
  deleteCheckpoint, isCompatible, loadCheckpoint, makeCheckpoint, saveCheckpoint, storageErrorMessage,
  type Checkpoint,
} from './checkpoint';
import { decode, loadDataset } from './data';
import { MODEL_DEFAULTS, countParams, nextTokenLogits } from './model';
import { isWujue, pushSample, samplePoem, type SampleRecord } from './sample';
import { Trainer, type TrainState, type TrainStats } from './train';
import { drawLossChart } from './ui/lossChart';
import { renderSamples } from './ui/samples';

const $ = (testId: string) => document.querySelector<HTMLElement>(`[data-testid="${testId}"]`)!;
const startToggle = $('start-toggle') as HTMLButtonElement;
const lossChart = $('loss-chart') as HTMLCanvasElement;
const temperatureInput = $('temperature') as HTMLInputElement;
const constrainBox = $('constrain') as HTMLInputElement;
const generateBtn = $('generate') as HTMLButtonElement;

// 浏览器刷新时可能恢复滑块位置，所以一开始就按滑块的实际值显示一次
const showTemperature = () => ($('temperature-value').textContent = Number(temperatureInput.value).toFixed(1));
showTemperature();
temperatureInput.addEventListener('input', showTemperature);

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

let samples: SampleRecord[] = [];
renderSamples($('samples'), samples);

Promise.all([initBackend(location.search), loadDataset()])
  .then(async ([be, ds]) => {
    const V = ds.vocab.length;
    const tokenIds = { bos: ds.bosId, comma: ds.stoi.get('，')!, period: ds.stoi.get('。')! };
    $('dataset-info').textContent = `数据集：${ds.poems.length} 首五言绝句 · 字表 ${V} 字 · 来自 chinese-poetry（MIT）`;
    $('loss-baseline').textContent = Math.log(V).toFixed(3);
    $('backend').textContent = BACKEND_LABEL[be.backend];
    const notice = $('backend-notice');
    notice.textContent = be.notices.join('\n');
    notice.hidden = be.notices.length === 0;

    // 用 requestAnimationFrame 合并重绘：每帧最多画一次
    let drawQueued = false;
    const queueDraw = () => {
      if (drawQueued) return;
      drawQueued = true;
      requestAnimationFrame(() => {
        drawQueued = false;
        $('loss-empty').hidden = trainer.lossHistory.length + trainer.valHistory.length > 0;
        drawLossChart(lossChart, trainer.lossHistory, trainer.valHistory);
      });
    };

    // 用控件上当前的温度和格式开关写一首诗，加进生成记录
    const generate = async (step: number, valLoss: number | null, manual: boolean) => {
      const temperature = Number(temperatureInput.value);
      const constrained = constrainBox.checked;
      const poem = await samplePoem((s) => nextTokenLogits(trainer.params, trainer.cfg, s), tokenIds, {
        temperature,
        constrain: constrained,
      });
      const text = decode(poem, ds.vocab);
      samples = pushSample(samples, { step, valLoss, temperature, constrained, manual, text, formatOk: isWujue(text) });
      renderSamples($('samples'), samples);
    };

    const trainer = new Trainer({
      cfg: { ...MODEL_DEFAULTS, vocabSize: V },
      train: ds.train,
      val: ds.val,
      bosId: ds.bosId,
      onStep: (s) => {
        renderStats(s);
        queueDraw();
      },
      onMilestone: async (m) => {
        $('loss-val').textContent = fmtLoss(m.valLoss);
        queueDraw();
        // 训练循环会等这首诗写完再走下一步，采样不会和训练并发读写权重
        await generate(m.step, m.valLoss, false);
      },
      onState: (s) => (startToggle.textContent = TOGGLE_TEXT[s]),
    });
    queueDraw(); // 先按实际尺寸画一次空画布
    addEventListener('resize', queueDraw);
    // 里程碑（验证、采样）期间没有 onStep，「已训练」也要继续走字
    setInterval(() => trainer.state === 'running' && renderStats(trainer.stats()), 500);
    $('params').textContent = (countParams(trainer.params) / 1e6).toFixed(2) + 'M';

    const saveBtn = $('save') as HTMLButtonElement;
    const loadBtn = $('load') as HTMLButtonElement;
    const deleteBtn = $('delete-save') as HTMLButtonElement;
    const ignoreBtn = $('ignore-save') as HTMLButtonElement;
    const saveStatus = $('save-status');
    let checkpoint: Checkpoint | null = null;
    let compatible = false;

    const updateSaveButtons = () => {
      saveBtn.disabled = checkpoint !== null && !compatible;
      loadBtn.disabled = !compatible;
      deleteBtn.disabled = checkpoint === null;
    };
    const runSaveAction = async (action: () => Promise<void>, errorMessage = storageErrorMessage) => {
      saveBtn.disabled = loadBtn.disabled = deleteBtn.disabled = true;
      ignoreBtn.hidden = true;
      try {
        await action();
      } catch (error) {
        saveStatus.textContent = errorMessage(error);
      } finally {
        updateSaveButtons();
      }
    };

    try {
      checkpoint = await loadCheckpoint();
      if (checkpoint) {
        compatible = isCompatible(checkpoint, ds.hash, trainer.cfg);
        saveStatus.textContent = compatible
          ? `发现存档：第 ${checkpoint.step} 步（${new Date(checkpoint.savedAt).toLocaleString('zh-CN')}）`
          : '存档与当前数据集或模型不兼容';
        ignoreBtn.hidden = !compatible;
      }
    } catch (error) {
      saveStatus.textContent = `无法读取存档：${error instanceof Error ? error.message : String(error)}`;
    }
    updateSaveButtons();

    ignoreBtn.addEventListener('click', () => {
      ignoreBtn.hidden = true;
      saveStatus.textContent = '';
    });
    saveBtn.addEventListener('click', () => runSaveAction(async () => {
      saveStatus.textContent = '正在保存…';
      const next = await trainer.runBetweenSteps(async () =>
        makeCheckpoint(await trainer.snapshot(), ds.hash, trainer.cfg, samples),
      );
      await saveCheckpoint(next);
      checkpoint = next;
      compatible = true;
      saveStatus.textContent = `已保存：第 ${next.step} 步`;
    }));
    loadBtn.addEventListener('click', () => {
      const saved = checkpoint;
      if (!saved || !compatible) return;
      return runSaveAction(async () => {
        startToggle.disabled = generateBtn.disabled = true;
        try {
          await trainer.restore(saved);
          samples = [...saved.samples];
          renderSamples($('samples'), samples);
          renderStats(trainer.stats());
          $('loss-val').textContent = fmtLoss(trainer.valHistory.at(-1)?.loss ?? null);
          queueDraw();
          saveStatus.textContent = `已载入：第 ${saved.step} 步`;
        } finally {
          startToggle.disabled = generateBtn.disabled = false;
        }
      }, (error) => `无法载入存档：${error instanceof Error ? error.message : String(error)}`);
    });
    deleteBtn.addEventListener('click', () => {
      if (!confirm('确定删除存档？')) return;
      return runSaveAction(async () => {
        await deleteCheckpoint();
        checkpoint = null;
        compatible = false;
        saveStatus.textContent = '存档已删除';
      }, (error) => `无法删除存档：${error instanceof Error ? error.message : String(error)}`);
    });

    // 暂停要等当前这一步结束，按钮文字由 onState 在循环真正停下后更新
    startToggle.addEventListener('click', () => (trainer.state === 'running' ? trainer.pause() : trainer.start()));
    startToggle.textContent = TOGGLE_TEXT[trainer.state];
    startToggle.disabled = false;

    // 训练中排到当前这一步之后执行，训练不中断；步数和验证 loss 也在那时才读
    generateBtn.addEventListener('click', () =>
      trainer.runBetweenSteps(() => generate(trainer.lossHistory.length, trainer.valHistory.at(-1)?.loss ?? null, true)),
    );
    generateBtn.disabled = false;
  })
  .catch((err) => {
    // ponytail: 只给出文案；「重试」按钮在 Task 13 加
    console.error(err);
    $('dataset-info').textContent = '数据集加载失败';
  });
