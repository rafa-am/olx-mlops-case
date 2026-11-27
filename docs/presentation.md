# mlt — Roteiro de Apresentação (30 min)

Estrutura: 30 min de apresentação + 30 min de perguntas e discussão.

---

## 1. Abertura (2 min)

**Frase de abertura:**
> "O cientista entrega `model.py`, `requirements.txt` e dados. O `mlt` faz o resto: ambiente isolado, rastreamento, reprodutibilidade e API."

**O que cobrir:**
- O problema: cientistas escrevem Python, não Dockerfile, não servidor, não infraestrutura.
- A solução: contrato mínimo (`MLModel`, dois arquivos obrigatórios) e uma CLI que gerencia o ciclo completo.
- O que *não* é: não é MLflow, não é uma plataforma, não é um servidor de modelos compartilhado. É uma ferramenta local que tem caminho claro para produção.

---

## 2. Demo ao vivo (12 min)

Executar na ordem. Ter um terminal limpo com `mlt` instalado e nenhum projeto em `/tmp/demo` antes de começar.

```bash
# 1. Criar projeto (1 min)
mlt init /tmp/demo --template iris
cd /tmp/demo

# 2. Validar contrato sem treinar (1 min)
mlt check .
# Mostrar: python, runner, seed — os valores efetivos e suas origens

# 3. Treinar (3 min)
mlt train .
# Mostrar: building environment... training... run_id, status, accuracy
# Ponto de fala: imagem cacheada em mlt-env:<hash>; próximo treino = segundos

# 4. Treinar de novo sem mudar nada (1 min)
mlt train .
# Mostrar: retorna imediatamente com o mesmo run_id — idempotência pelo fingerprint

# 5. Inspecionar o run (1 min)
mlt runs list
mlt runs show latest
# Mostrar: seções identity / inputs / environment / outputs

# 6. Servir o modelo (3 min)
mlt serve latest
# Em outro terminal:
curl -s -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"inputs": [{"sepal_length": 5.1, "sepal_width": 3.5,
                   "petal_length": 1.4, "petal_width": 0.2}]}' \
  | python3 -m json.tool
# Mostrar: {"predictions": ["setosa"], "run_id": "..."}
# Mencionar: /docs (Swagger), /schema, /health
```

**Pontos de fala durante a demo:**
- O URL só é impresso depois que `/health` retorna 200 — nunca há 503 imediato.
- `latest` é um alias; qualquer prefixo único do run_id funciona (como `git`).
- O servidor roda dentro do ambiente isolado, com as dependências do cientista — não no host.

---

## 3. Arquitetura (8 min)

### Local MVP (3 min)

Mostrar o diagrama de componentes de `docs/architecture.md`:

```
CLI (thin layer) → mlt.core → Runner (Docker/venv)
                            → RunStore (files + SQLite)
                            → DataSource (local folder)
```

**Decisão para defender:** por que separar `Runner`, `RunStore` e `DataSource` em interfaces?
> Cada interface tem uma implementação local no MVP e um caminho claro para o cloud. Trocar `RunStore` por S3 + DynamoDB não muda o contrato do cientista nem o formato do manifest.

**Decisão para defender:** por que o manifest é o source of truth e não o banco?
> O SQLite é um índice rebuildável. O manifest é auto-suficiente: um run produzido localmente e um produzido na AWS são o mesmo JSON, e `diff`/`verify` funcionam nos dois.

### Produção na AWS (5 min)

Mostrar o diagrama AWS de `docs/architecture.md`:

```
Scientist CLI → Control API (API GW + Lambda)
             → Training (AWS Batch, no internet, VPC endpoints)
             → S3 (artifacts) + DynamoDB (RunStore index)
             → Inference (ECS Fargate + ALB)
```

**Mapeamento MVP → AWS:**
| MVP | AWS |
|-----|-----|
| VenvRunner / DockerRunner | AWS Batch job |
| RunStore (files + SQLite) | S3 + DynamoDB |
| `mlt serve` (foreground) | ECS Fargate + ALB |
| LocalDataSource | S3 + Iceberg (snapshots) |

**Por que Batch e não SageMaker Training?**
> SageMaker impõe opiniões sobre entry points e data channels que conflitam com o modelo arbitrário do `mlt`. Batch roda qualquer imagem Docker sem restrições de framework.

---

## 4. Reprodutibilidade (4 min)

### Fingerprint (1 min)

```
fingerprint = hash(code + params + schema + deps + python + seed)
```

- Runner **fora** do fingerprint: mesmo projeto em Docker e venv → mesmo fingerprint.
- `params.yaml` e `schema.yaml` têm componentes próprios no fingerprint — não entram no hash do código (bug corrigido durante o desenvolvimento: mudança de params aparecia como mudança de código no `diff`).

### diff (1 min)

```bash
mlt diff <run_a> <run_b>
# Mostra só o que mudou: params, código, dados, seed, deps, métricas com delta
```

### verify (2 min)

```bash
mlt verify <run_id>
# Retreina a partir do snapshot code/ do run (nunca do projeto raiz)
# Cria run com verification_of: <run_id>
# Exit 0 = reproduzido; exit 4 = divergência nas métricas
```

**O que prova e o que não prova:**
- Prova: mesmo código + params + seed + deps → métricas equivalentes (tolerância 1e-6).
- Não prova: conteúdo idêntico dos dados sem `--snapshot-data` (identity-level, não content-level).

---

## 5. Encerramento + Web UI (4 min)

### Web UI (2 min)

```bash
mlt ui /tmp/demo
```

Abrir no browser. Mostrar:
- Home: status do projeto, próximo passo sugerido.
- Runs: lista com badges de status, `↻ verifying` quando verify está rodando.
- **Compare (diff):** selecionar dois runs, mostrar o que mudou — mais legível que no terminal.
- **Verify:** iniciar verificação, mostrar step-by-step ao vivo.

**Ponto de fala:**
> "A UI chama as mesmas funções de `mlt.core` que a CLI. Isso só foi possível porque o núcleo foi isolado — sem `click`, sem `typer`, sem prints. A UI é experimental (sem testes automatizados), mas valida a arquitetura."

### O que ficou fora (P2) e próximos passos (1 min)

P2 não implementado (por escolha, não por falta de tempo):
- `--snapshot-data`: snapshot content-addressed dos dados para reprodutibilidade total.
- `serve --detach` + `ps` / `stop` / `logs`: serving em background.
- `blank` template: skeleton para frameworks além de scikit-learn.

Próximos passos de evolução (em ordem):
1. Isolar o runtime do `mlt` do ambiente do cientista (FastAPI/uvicorn no env do cientista = risco de conflito de versões).
2. `DataSource` remoto (S3 + Iceberg).
3. CLI como cliente de uma Control API; Runner no AWS Batch.

### Fechamento (1 min)

> "O foco foi: contrato mínimo, runs imutáveis, reprodutibilidade verificável e um caminho claro para produção. Cada decisão tem uma alternativa descartada no `DECISIONS.md`."

---

## 6. Perguntas prováveis

**Por que não MLflow?**
> MLflow é pesado para o escopo — requer servidor, banco, UI própria. A interface `RunStore` permite adotá-lo como backend mais tarde sem mudar o contrato do cientista.

**Como escala para múltiplos usuários / times?**
> Localmente, cada cientista tem seu próprio `.mlt/`. Em produção, a Control API centraliza os runs em S3 + DynamoDB; o manifest é o mesmo formato, `diff` e `verify` funcionam entre máquinas.

**Como lida com modelos grandes (>GB)?**
> `save` e `load` operam em um diretório local criado pelo `mlt` — o cientista escreve o que quiser lá. O `RunStore` é responsável pelo transporte; com S3 seria um upload multipart transparente. Hoje a limitação é o disco local.

**Por que Docker e não só venv?**
> Venv não isola rede nem pacotes de sistema. Docker garante `--network none` durante o treino e um ambiente bit-a-bit reproduzível. Venv é a alternativa para quem não tem Docker — com isolamento reduzido gravado no manifest.

**O que acontece se `predict` lançar exceção?**
> O servidor retorna 500 sem stack trace no corpo (não vazar informação). O erro vai para o log do servidor. O modelo continua servindo as próximas requisições.

**Como o cientista sabe qual modelo está servindo?**
> `GET /metadata` retorna `run_id`, `fingerprint`, métricas e hash do schema. `GET /schema` retorna o schema efetivo do run servido — nunca do projeto raiz.

**Por que manifest e não só banco?**
> O manifest é self-contained: um arquivo JSON por run que carrega tudo para `diff` e `verify` funcionarem sem banco. O SQLite é um índice rebuildável. Em produção, o DynamoDB substitui o SQLite mas o manifest permanece o source of truth — o mesmo arquivo local e na AWS.

**Por que separar `params.yaml` do código?**
> Configuração lida sem executar código é mais fácil de revisar, versionar e comparar entre runs. Mantém o contrato menor — `model.py` não precisa saber de hiperparâmetros hardcoded.

**O cientista precisa mudar muito do código dele para usar o `mlt`?**
> Precisa herdar de `MLModel` (ou `SklearnModel`) e implementar `fit`. Para scikit-learn, é só `fit` atribuindo `self.model`. O `mlt init --template iris` entrega o esqueleto pronto. A fricção de herança é o trade-off consciente para ter um contrato verificável.
