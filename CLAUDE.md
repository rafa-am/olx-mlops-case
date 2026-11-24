# CLAUDE.md

Contexto do projeto para o Claude Code.

## O que é

`mlt`: ferramenta (pacote Python com CLI) que leva um cientista de dados do treino reprodutível e isolado até uma API REST de inferência, a partir de apenas código Python. É um case técnico para uma vaga de Machine Learning Engineer. O foco é a ferramenta, não o modelo.

Documentos de referência, em ordem de autoridade:

1. `docs/design.md`: o que construir (contratos, manifesto, API, códigos de saída, ordem de construção).
2. `DECISIONS.md`: por que cada decisão foi tomada e as alternativas descartadas.
3. `AI_USAGE.md`: registro do uso de IA (ver regra abaixo).

Se algo no código contradiz o `docs/design.md`, pare e pergunte antes de seguir. Se uma decisão mudar na implementação, atualize `docs/design.md` e `DECISIONS.md` no mesmo commit.

## Regras de trabalho

- Trabalhe uma etapa por vez, na ordem de `docs/design.md` seção 16. Planeje antes de cada etapa e espere a revisão do plano.
- Respeite as prioridades P0/P1/P2 da seção 16: faça P0 primeiro; P2 só se sobrar tempo. Registre o tempo gasto por etapa e o que foi cortado.
- O run guarda `code/` (cópia do projeto sem `data/` e `.mlt/`) e `params.yaml`; o `verify` retreina a partir de `code/`. O `mlt` entra na imagem Docker copiado para o contexto de build (hash inclui o código-fonte do `mlt`).
- Commits pequenos, um por etapa ou comportamento, com os testes passando antes de cada um. Mensagens em português, no imperativo, curtas.
- Escreva os testes dos fluxos críticos (seção 15) junto com cada etapa, não depois.
- Não adicione dependências nem recursos fora do escopo (seção 17) sem perguntar.
- Código simples e legível: o autor vai defender cada decisão por 30 minutos. Explique escolhas não óbvias em comentário curto.
- Nada de segredos no repositório. `.mlt/`, `~/.mlt`, caches e dados gerados ficam no `.gitignore`.

## Regras de design que não podem ser violadas

- A CLI é só camada fina: a lógica fica em `mlt.core`. Os códigos de saída são uma enumeração do núcleo.
- O núcleo de `src/mlt/model/` (`MLModel`, com `fit`, `predict`, `save` e `load` abstratos) não importa nada fora da biblioteca padrão e não traz padrões ligados a framework. Conveniências como `SklearnModel` (joblib e pandas) ficam em módulos próprios, com importação preguiçosa. O template do Iris herda de `SklearnModel`.
- O treino roda sem rede e com dados somente leitura. Os dados nunca entram na imagem Docker.
- O run é imutável. O manifesto é a fonte de verdade; o SQLite é só índice reconstruível.
- O fingerprint é calculado só sobre `inputs`. O hash do código ignora `.mlt/`, `.git/`, `__pycache__/` e dados.
- Raiz é insumo, run é registro: `serve`, `verify` e `diff` leem do run; `check` lê da raiz.
- O serving obtém o run sempre pelo `RunStore`, nunca lendo `runs/` diretamente.
- O `check` é estático (não importa o `model.py` no host). A validação profunda roda dentro do ambiente, antes de criar o run.
- `save` e `load` operam sobre um diretório local; o transporte para um registry remoto é do `RunStore`, nunca do modelo.
- Erro de contrato (código 3) não cria run. Erro de execução (4) e timeout (5) criam run `failed`.
- `.mlt/` fica dentro do projeto do cientista, nunca no diretório atual por acaso.
- O README traz o guia do cientista (uma página, ver `docs/design.md` seção 4): classe base a escolher, o que `fit` e `predict` recebem, `requirements.txt` completo e dados em `<projeto>/data/` ou `--data`. Os templates `iris` e `blank` têm comentários curtos em cada método.
- Não prometer o que não se garante: sem `--snapshot-data`, a rastreabilidade dos dados é de identidade, não de conteúdo.

## Estrutura

```
src/mlt/{cli,core,model,runners,datasources,runstore,serving,templates/{iris,blank}}
tests/
docs/{design.md,architecture.md}
README.md  DECISIONS.md  AI_USAGE.md
```

## Comandos

- Instalar para desenvolvimento: `pip install -e ".[dev]"`
- Testes: `pytest` (testes com Docker têm a marca `docker` e são pulados sem Docker; `pytest -m docker` roda só eles)
- Exemplo ponta a ponta: `mlt init /tmp/demo --template iris && mlt train /tmp/demo && mlt runs list`
- Formatação e lint: a definir na etapa 1 (sugestão: ruff).

## Registro do uso de IA (obrigatório)

Ao final de cada etapa, acrescente ao `AI_USAGE.md` (seção "O que a IA gerou e foi corrigido") qualquer ponto em que o autor corrigiu, rejeitou ou ajustou o que foi gerado, com o motivo. Se não houve correção, diga isso na resposta, para o autor registrar o que verificou e como.

## Fora do escopo

Autenticação, versionamento de dados, rollback e A/B, monitoramento, GPU, treino distribuído, UI, limites de CPU e memória, limpeza de runs, versionamento de rota, estatísticas de dados no manifesto. A nuvem (AWS) é só desenho.

- Instalação: Python `>=3.10` no host (`pyproject.toml`). `mlt --version` mostra versão, Python e a detecção do Docker (disponível e daemon ativo). O README lista como pré-requisitos Docker em execução (opcional) e rede no primeiro build.
- `mlt init <pasta>`: cria pasta nova, recusa pasta não vazia (saída 2), gera `.gitignore` com `.mlt/`, não roda `git init`, imprime os próximos passos.
- `check`: dados via `DataSource.check()` (MVP: pasta local existe e não vazia; erro saída 3); herança indireta gera aviso, não erro; saída `ok`/`aviso`/`erro` com arquivo e linha. Descoberta de projeto sobe até `model.py` ou `.mlt/`; `runs list` sem runs responde saída 0.
- `train`: resumo final com run_id, status, métricas e próxima ação; aviso de build lento; run idêntico não treina (saída 0, `--force` repete); Ctrl+C marca run `failed` (`interrupted`) e sai 130; falha de build sai 6 sem criar run.
- Python: precedência `--python` > `.python-version` > padrão (Docker 3.12; venv usa `sys.executable`, sem procurar no PATH; versão diferente no venv sai 6). Major.minor entra no `deps` do fingerprint. Runner fora do fingerprint (só em `environment`). `train` e `check` imprimem valores efetivos e a origem (python, runner, seed). Sem Docker: `auto` cai para venv com aviso, `docker` explícito sai 6.
- `runs list`: linha por run (run_id, status, quando, duração, até 3 métricas, fingerprint curto), limite 20, `--all`. `runs show`: seções identidade/entradas/ambiente/resultados/erro; `--json` = manifesto. Todo comando com `run_id` aceita `latest` e prefixo único; inexistente ou ambíguo sai 7.
- `serve`: imprime URL, run_id, /docs e `curl` preenchido só após o modelo carregar e /health 200; porta padrão 8000, ocupada sai 6 (sugere --port, não troca sozinho); /docs mantido; schema lido do run, nunca da raiz.
- `diff <a> <b>`: lê manifestos, começa com 'fingerprint: diferente (componentes)', mostra só o que difere. `verify <run_id>`: versão mínima; retreina só do que o run guardou (`code/`, params, schema, seed, Python) com o mesmo runner; cria run com `verification_of`; compara fingerprint e métricas (tolerância relativa fixa 1e-6); saída 0 reproduzido, 4 divergiu, 8 se dados mudaram sem snapshot. Sem `--tolerance` nem hash de artefato. Cortável: o teste de integração 'treina duas vezes' cobre o argumento.
- P2 com fluxo fechado em `docs/design.md` (antes da seção 17): `--snapshot-data` (store por conteúdo, `frozen`), `serve --detach` + `ps`/`stop`/`logs` (PID em `.mlt/serving/`), reconciliação por `flock`, template `blank` (marcador `# TODO(mlt)`), `--json` (list, show, diff, verify), `runs list --status`. `init --force` não será feito. `serve --detach` é P2: o P0 é primeiro plano.
