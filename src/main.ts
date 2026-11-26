import './style.css';
import { BACKEND_LABEL, initBackend } from './backend';
import {
  deleteCheckpoint, isCompatible, loadCheckpoint, makeCheckpoint, saveCheckpoint, storageErrorMessage,
  type Checkpoint,
} from './checkpoint';
import { BOS, decode, encode, loadDataset, validateCustomPoem, type Dataset } from './data';
import { MODEL_DEFAULTS, attentionMaps, countParams, nextTokenLogits } from './model';
import presets from './presets.json';
import { isWujue, pushSample, samplePoem, type SampleRecord } from './sample';
import { BATCH_SIZE, CPU_BATCH_SIZE, Trainer, type TrainState, type TrainStats } from './train';
import { createHeatmap } from './ui/heatmap';
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

// 后端先就绪就先显示徽章，不用等数据
const backendReady = initBackend(location.search).then((be) => {
  $('backend').textContent = BACKEND_LABEL[be.backend];
  const notice = $('backend-notice');
  notice.textContent = be.notices.join('\n');
  notice.hidden = be.notices.length === 0;
  return be.backend;
});

/** 加载失败时显示「重试」，点一次再加载一次，直到成功 */
function loadDatasetWithRetry(): Promise<Dataset> {
  return loadDataset().catch((err) => {
    console.error(err);
    $('dataset-info').hidden = true;
    $('data-error').hidden = false;
    return new Promise<Dataset>((resolve) =>
      $('retry').addEventListener('click', () => {
        $('data-error').hidden = true;
        $('dataset-info').hidden = false;
        resolve(loadDatasetWithRetry());
      }, { once: true }),
    );
  });
}

Promise.all([backendReady, loadDatasetWithRetry()])
  .then(async ([backend, ds]) => {
    const V = ds.vocab.length;
    const tokenIds = { bos: ds.bosId, comma: ds.stoi.get('，')!, period: ds.stoi.get('。')! };
    $('dataset-info').textContent = `数据集：${ds.poems.length} 首五言绝句 · 字表 ${V} 字 · 来自 chinese-poetry（MIT）`;
    $('loss-baseline').textContent = Math.log(V).toFixed(3);

    const heatmapSelect = $('heatmap-select') as HTMLSelectElement;
    const latest = new Option('最新生成的诗', 'latest');
    latest.disabled = true;
    heatmapSelect.replaceChildren(...presets.map(({ title, poem }) => new Option(title, poem)), latest);
    heatmapSelect.selectedIndex = 0;
    const heatmap = createHeatmap($('heatmap'));
    const syncLatestPoem = () => {
      latest.disabled = samples.length === 0;
      if (latest.disabled && latest.selected) heatmapSelect.selectedIndex = 0;
    };
    const refreshHeatmap = async () => {
      const step = trainer.lossHistory.length;
      const poem = latest.selected ? samples[0].text : heatmapSelect.value;
      const prefix = [...poem].slice(0, trainer.cfg.ctx - 1);
      const ids = [ds.bosId, ...encode(prefix.join(''), ds.stoi)];
      heatmap.update(await attentionMaps(trainer.params, trainer.cfg, ids), [BOS, ...prefix], step);
    };

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
      syncLatestPoem();
      if (manual && latest.selected) await refreshHeatmap();
    };

    const trainer = new Trainer({
      cfg: { ...MODEL_DEFAULTS, vocabSize: V },
      train: ds.train,
      val: ds.val,
      bosId: ds.bosId,
      batchSize: backend === 'cpu' ? CPU_BATCH_SIZE : BATCH_SIZE,
      onStep: (s) => {
        renderStats(s);
        queueDraw();
      },
      onMilestone: async (m) => {
        $('loss-val').textContent = fmtLoss(m.valLoss);
        queueDraw();
        // 训练循环会等这首诗写完再走下一步，采样不会和训练并发读写权重
        await generate(m.step, m.valLoss, false);
        await refreshHeatmap();
      },
      onState: (s) => (startToggle.textContent = TOGGLE_TEXT[s]),
    });
    queueDraw(); // 先按实际尺寸画一次空画布
    addEventListener('resize', queueDraw);
    // 里程碑（验证、采样）期间没有 onStep，「已训练」也要继续走字
    setInterval(() => trainer.state === 'running' && renderStats(trainer.stats()), 500);
    $('params').textContent = (countParams(trainer.params) / 1e6).toFixed(2) + 'M';
    await refreshHeatmap();
    heatmapSelect.addEventListener('change', () => trainer.runBetweenSteps(refreshHeatmap));
    // 校验通过才换诗；「自定义」选项第一次成功时加到下拉框末尾，之后只改它的诗
    const heatmapInput = $('heatmap-input') as HTMLInputElement;
    const viewBtn = $('heatmap-view') as HTMLButtonElement;
    const custom = new Option('自定义');
    heatmapInput.form!.addEventListener('submit', (e) => {
      e.preventDefault();
      const result = validateCustomPoem(heatmapInput.value, ds.stoi);
      $('heatmap-error').textContent = result.ok ? '' : result.error;
      if (!result.ok) return;
      custom.value = result.poem;
      heatmapSelect.add(custom);
      custom.selected = true;
      trainer.runBetweenSteps(refreshHeatmap);
    });
    heatmapSelect.disabled = heatmapInput.disabled = viewBtn.disabled = false;

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
          syncLatestPoem();
          await trainer.runBetweenSteps(refreshHeatmap);
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
    // 数据加载失败由「重试」处理；到这里只剩后端全部失败（cpu 实际总能用）或初始化代码出错
    console.error(err);
    $('dataset-info').textContent = `初始化失败：${err instanceof Error ? err.message : String(err)}`;
  });
