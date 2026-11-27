export function setupExplainer(open: HTMLButtonElement, dialog: HTMLDialogElement): void {
  const video = dialog.querySelector('video')!;
  const error = dialog.querySelector<HTMLElement>('[data-testid="explainer-error"]')!;
  const close = dialog.querySelector<HTMLButtonElement>('[data-testid="explainer-close"]')!;

  video.addEventListener('error', () => { error.hidden = false; });
  open.addEventListener('click', () => {
    dialog.showModal();
    if (!video.hasAttribute('src')) video.src = 'explainer.mp4';
  });
  close.addEventListener('click', () => dialog.close());
  dialog.addEventListener('close', () => {
    video.pause();
    open.focus();
  });
}
