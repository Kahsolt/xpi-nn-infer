from typing import Callable, Generator, Any

# (inputs, outputs, index) -> None
Callback = Callable[[Any, Any, int], None]


class Inferer:

  model: Any

  def infer(self, x:Any) -> Any:
    return self.model(x)

  def infer_stream(self, datagen:Generator[Any, None, None], callback:Callback=None):
    for i, x in enumerate(datagen):
      o = self.infer(x)
      if callback:
        callback(x, o, i)

  def benchmark(self, inputs:Any, callback:Callback=None, n_iter:int=3000, log_every:int=100) -> float:
    from time import perf_counter
    from xpi_nn_infer.utils import ValueWindow

    fps_list: list[float] = []
    vw = ValueWindow(log_every * 2)
    for i in range(1, 1 + n_iter):
      ts_start = perf_counter()
      outputs = self.infer(inputs)
      if callback:
        callback(inputs, outputs, i)
      ts_end = perf_counter()
      if i >= 500:   # warm-ups
        vw.append(ts_end - ts_start)

      if i % log_every == 0 and vw.sum:
        fps = 1 / vw.mean
        fps_list.append(fps)
        print('>> FPS:', fps)

    fps_avg = (sum(fps_list) / len(fps_list)) if len(fps_list) else -1
    print('>> Mean FPS:', fps_avg)
    return fps_avg
