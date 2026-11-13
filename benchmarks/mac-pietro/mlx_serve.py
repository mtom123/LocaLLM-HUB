"""mlx_lm.server con qualsiasi nome di modello mappato sul modello caricato (OMP e i banchi mandano "agents-a1", "x", ...).
Uso: python mlx_serve.py --model <cartella MLX> --port 8080 --max-tokens 8192 [--chat-template-args {"enable_thinking":false}]"""
import mlx_lm.server as S
_orig = S.ModelProvider.load
def load(self, model_path, adapter_path=None, draft_model_path=None):
    if model_path not in self._model_map:
        model_path = "default_model"
    if draft_model_path not in self._draft_model_map:
        draft_model_path = "default_model"
    return _orig(self, model_path, adapter_path, draft_model_path)
S.ModelProvider.load = load
if __name__ == "__main__":
    S.main()
