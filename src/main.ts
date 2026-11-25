import './style.css';
import { loadDataset } from './data';

const $ = (testId: string) => document.querySelector<HTMLElement>(`[data-testid="${testId}"]`)!;
const startToggle = $('start-toggle') as HTMLButtonElement;

loadDataset()
  .then((ds) => {
    const V = ds.vocab.length;
    $('dataset-info').textContent = `数据集：${ds.poems.length} 首五言绝句 · 字表 ${V} 字 · 来自 chinese-poetry（MIT）`;
    $('loss-baseline').textContent = Math.log(V).toFixed(3);
    startToggle.textContent = '开始训练';
    startToggle.disabled = false;
  })
  .catch((err) => {
    // ponytail: 只给出文案；「重试」按钮在 Task 13 加
    console.error(err);
    $('dataset-info').textContent = '数据集加载失败';
  });
