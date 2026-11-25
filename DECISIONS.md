# DECISIONS.md: Case MLOps

## Visão geral

O `mlt` é um pacote Python com CLI que leva o cientista de dados do treino reprodutível à API REST, a partir de apenas código Python.

O cientista fornece `requirements.txt`, `model.py` (uma classe que herda de `MLModel`), dados e, opcionalmente, `params.yaml` e `schema.yaml`. A ferramenta cuida de isolamento, registro e serving.

1. `mlt init` cria um projeto a partir de um modelo (o Iris é o exemplo).
2. `mlt check` valida o contrato sem treinar.
3. `mlt train` executa o treino em Docker (principal) ou venv, sem rede, e grava um run com modelo, métricas e manifesto.
4. `mlt runs list` e `mlt runs show` identificam cada run.
5. `mlt serve` sobe uma API FastAPI para um run específico.

A CLI é uma camada fina sobre `mlt.core`. `DataSource`, `Runner` e `RunStore` são interfaces com implementação local no MVP e caminho de evolução para a nuvem.

## Contrato do cientista

O contrato mínimo é uma pasta com `requirements.txt` e `model.py`. `params.yaml`, `schema.yaml` e dados são opcionais ou externos.

- `model.py` define exatamente uma subclasse de `MLModel`, com `fit(data_dir, params)` (devolve um dict de métricas) e `predict(inputs)` (lista de registros na entrada, lista de predições na saída).
- `save` e `load` também são abstratos no núcleo. A classe opcional `SklearnModel` traz `predict` padrão e `save` e `load` em joblib, e o cientista só escreve o `fit` e atribui `self.model`.
- A seed chega em `self.seed`; o cientista a usa nos estimadores.
- `params.yaml`, se existir, vira o argumento `params`. `schema.yaml`, se existir, declara as features de entrada; sem ele, o schema é inferido de `feature_names_in_`.

**Versão do Python:** o `requirements.txt` não consegue declará-la, então há um arquivo opcional `.python-version` (uma linha, convenção de pyenv e uv). Precedência: `--python`, depois o arquivo, depois o padrão.

**Por que classe e não função:** uma base comum dá reuso, um ponto único para verificar o contrato e `predict` garantido. A função seria mais simples para scripts curtos, mas espalharia as convenções.

**Por que o núcleo é agnóstico de framework:** um `predict` padrão baseado em Pipeline do scikit-learn no núcleo acoplaria a ferramenta a um tipo de artefato e dificultaria outros, como modelos treinados com PySpark. O núcleo exige `fit`, `predict`, `save` e `load`, e a ergonomia do scikit-learn vem na `SklearnModel`, opcional. A alternativa descartada foi pôr o `predict` padrão e o joblib no `MLModel`.

**Limite conhecido:** o `fit(data_dir, params)` assume dados em pasta local. Um modelo Spark pediria um `DataSource` que entregue uma URI e um `Runner` de cluster. A ferramenta poderia passar ao `fit` só os parâmetros que ele declara (como as fixtures do pytest), mas isso não é implementado no MVP.

**Por que schema em YAML e não em Pydantic na classe:** tira a definição do código, permite revisar e versionar o que foi inferido e mantém o contrato menor. Perde validações cruzadas entre campos, o que fica registrado como limitação.

## Isolamento e reprodutibilidade

Cada treino roda em um ambiente isolado e identificado por hashes, e o run registra tudo o que o identifica.

- **Docker (principal):** imagem `mlt-env:<hash>`, com hash do `requirements.txt`, da versão do Python e da versão do `mlt`. A base é `python:3.12-slim` (`--python` troca). O build tem rede; o treino roda sem rede, com dados somente leitura.
- **venv (alternativa):** mesmo contrato, com cache em `~/.mlt/envs/<hash>`. Não isola rede nem sistema, e o manifesto registra o nível de isolamento.
- **Fingerprint:** hash de código, dados, params, schema declarado, dependências e seed. É a identidade do run.
- **Seed:** `--seed`, depois `MLT_SEED`, depois 42. O valor efetivo e a origem vão no manifesto.
- **Dependências:** após o build, o `pip freeze` é gravado como lock no manifesto. Requisitos sem versão fixa geram aviso.
- **Garantia:** mesmo fingerprint significa mesmos insumos identificados. O `verify` compara métricas com tolerância, porque a seed não elimina o não determinismo de threads e bibliotecas.

**Limite conhecido:** o runtime do `mlt` é instalado no mesmo ambiente das dependências do cientista, então versões fixadas podem conflitar com FastAPI ou Pydantic. O isolamento do runtime em dois ambientes ficou como evolução.

**Como o `mlt` entra na imagem:** o runner copia o pacote instalado para o contexto de build. Como a versão não muda durante o desenvolvimento, o hash do ambiente inclui também um hash do código-fonte do `mlt`, e `--rebuild` força a reconstrução.

**Rede e montagens:** o build precisa de rede uma vez (pacotes e imagem base); depois, com cache, o treino roda offline. No container, projeto e dados são somente leitura, a saída do run é gravável, os temporários vão em `tmpfs` e o usuário não é root.

**Python e runner:** no Docker, a base da imagem vem da versão pedida. No venv, o Python é o interpretador que executa o `mlt` (`sys.executable`), sem procurar outras versões no sistema; versão diferente da pedida sai com código 6, explicando como resolver. A versão major.minor entra no fingerprint, e o runner fica fora dele (só informativo), então o mesmo projeto em Docker e em venv tem o mesmo fingerprint. Sem Docker, o `auto` cai para o venv com aviso, e o manifesto registra `isolation: none`.

**`verify` e `diff`:** o `diff` compara dois manifestos e diz por que os fingerprints diferem. O `verify` é a prova prática de reprodutibilidade: retreina só a partir do que o run guardou (`code/`, params, schema, seed e Python), cria um run com `verification_of` e compara as métricas com tolerância relativa fixa de 1e-6 (saída 0 se reproduziu, 4 se divergiu). Não prova conteúdo igual dos dados sem snapshot nem reprodução bit a bit entre máquinas. Se faltar tempo, o comando é cortável e o mesmo argumento fica no teste de integração que treina duas vezes.

## Dados

Os dados ficam fora da imagem e entram no treino por uma interface `DataSource`, para que a origem possa mudar sem alterar o contrato.

- **MVP:** pasta local montada como volume somente leitura (`/data:ro`), sem cópia. O `fit` recebe `data_dir`.
- **Evolução:** S3 ou lakehouse (Iceberg, Parquet), materializados fora do sandbox, então o treino continua sem rede. Para volumes que não cabem em disco, leitura direta por VPC endpoint.
- **Rastreabilidade em níveis:** identificar (hash por arquivo e agregado, sempre), verificar (`mlt verify` recusa retreinar se os dados mudaram) e congelar (`--snapshot-data`, cópia endereçada por conteúdo).
- **Sem congelar**, a garantia é de identidade, não de conteúdo: dá para dizer se dois runs usaram os mesmos dados, não recuperá-los. Com lakehouse, o snapshot imutável resolve isso sem copiar.

Estatísticas por coluna no manifesto ajudariam a diagnosticar mudanças, mas só identificam; ficaram fora do MVP.

**Dados e `check`:** o `check` pergunta ao `DataSource` se a fonte é utilizável (no MVP, a pasta local existe e não está vazia; num lakehouse seria uma consulta de metadados). No treino, o `DataSource` materializa os dados em um diretório local somente leitura, fora do projeto e fora da imagem. Por isso o snapshot não precisa existir na pasta do cientista.

## Run, manifesto e RunStore

Cada treino gera um run imutável em `<projeto>/.mlt/runs/<run_id>/`, com modelo, `metrics.json`, `manifest.json`, `params.yaml` e `schema.yaml` efetivos, `code/` (cópia do projeto sem `data/` e `.mlt/`) e `train.log`. O run é autocontido: o `verify` retreina a partir de `code/`.

- **Manifesto (JSON):** blocos `inputs` (entram no fingerprint), `fingerprint`, `environment`, `outputs` e metadados do run, com `manifest_version`.
- **SQLite** é só índice de consulta (identidade, fingerprint, ambiente, métricas, caminhos). O manifesto é a fonte de verdade, e o índice é reconstruível.
- **Raiz é insumo, run é registro:** o treino copia `params.yaml` e `schema.yaml` para o run. `serve`, `verify` e `diff` leem do run, e `check` lê da raiz.
- **Âncora no projeto:** `.mlt/` fica dentro do projeto (reconhecido por `model.py`) e é ignorado no hash do código. Os demais comandos descobrem o projeto subindo diretórios, como o Git.
- **Concorrência:** `run_id` com sufixo aleatório e criação exclusiva, escrita atômica do manifesto, SQLite em modo WAL e lock por ambiente no build. Runs `running` sem processo vivo viram `failed` (`interrupted`).
- **Falhas:** erro de contrato não cria run. Erro de execução e timeout criam run `failed`, com `error` e log preservado.
- **Preparo para registry:** o run é um pacote autocontido, e o serving o obtém sempre pelo `RunStore`.

## API de inferência e serving

A API serve um run específico e tem quatro endpoints, com contrato mínimo e erros explícitos.

- **`POST /predict`:** entrada `{"inputs": [registros]}`, saída `{"predictions": [...], "run_id": "..."}`. Uma predição por registro, na mesma ordem, de forma livre e serializável em JSON. A validação é tudo ou nada: 422 (índice e campo), 400, 413, 500 ou 503.
- **`GET /schema`, `/metadata` e `/health`:** schema efetivo do run, identificação do modelo servido e prontidão.
- **Ciclo de vida:** `mlt serve <run_id>` em primeiro plano (ou `--detach`), `ps`, `stop` e `logs`. O alias `latest` vira o `run_id` concreto ao subir. Só serve run `succeeded`, e escuta em 127.0.0.1.
- **Imagem:** o ambiente do treino mais o servidor genérico. O modelo é montado somente leitura.

**Códigos de saída da CLI:** 0 sucesso, 1 erro inesperado, 2 uso incorreto, 3 contrato, 4 falha de execução, 5 timeout, 6 ambiente indisponível, 7 não encontrado, 8 operação recusada pelo estado.

Complementos: 130 para interrupção por Ctrl+C (o run fica `failed`, `interrupted`); 4 também para `verify` que diverge; 6 também para porta ocupada e para versão de Python indisponível no venv.

## Decisões de implementação (etapa 2)

**Run criado antes do worker rodar — exit 3 deleta, não marca failed.**
O design diz "erro de contrato dentro do ambiente não cria run". Na prática o run é criado em `running` antes de lançar o subprocess (para garantir consistência em caso de Ctrl+C). Se o worker sair com código 3 (ContractError na validação profunda), o trainer chama `store.remove_run()` e deleta o diretório — resultado visível: nenhum run no `runs list`. Se o worker sair com código 4 (ExecutionError no fit), o run é marcado `failed` e mantido (log preservado). Isso preserva o invariante do design sem requerer dois subprocessos (validate-only + train).

**Worker como subpacote `mlt._worker` (instalado no venv).**
Alternativa seria copiar um script avulso para o contexto de build. Usar um subpacote instalado junto com o `mlt` é mais limpo: sem arquivo temporário, sem caminho absoluto passado ao subprocess, e qualquer mudança no worker fica coberta pelo `mlt_source_hash` que invalida o cache do venv.

**VenvRunner usa `sys.executable` — sem busca no PATH.**
Se `.python-version` ou `--python` pedirem versão diferente do interpretador que roda o `mlt`, o trainer levanta `MltEnvironmentError` (exit 6) com instrução clara: usar Docker ou reinstalar o `mlt` num venv com aquela versão. Alternativa (procurar `python3.X` no PATH) tornaria o comportamento dependente do ambiente do host e difícil de testar.

## Decisões de implementação (etapa 3)

**Tratamento de erros na CLI — `_MltGroup.invoke` em vez de `main()`.**
A abordagem inicial capturava `MltError` somente em `main()`. Em testes via `CliRunner` que invocam `cli` diretamente, o erro não era capturado pelo Click e o `result.exit_code` vinha como 1 (exceção genérica) em vez do código correto. A solução foi criar uma subclasse `_MltGroup(click.Group)` que sobrescreve `invoke` e captura `MltError`, traduzindo para `ctx.exit(code)`. Com isso, o código de saída correto é retornado tanto na execução real (via `main()` em standalone_mode=True) quanto nos testes via CliRunner.

**Descoberta de projeto em dois modos.**
`train` e `check` recebem o caminho explicitamente e exigem `model.py` (erro 2 se ausente). `runs`, `diff` e `verify` sobem diretórios a partir do cwd até encontrar `model.py` ou `.mlt/`, com fallback para o cwd. Essa distinção reflete o design: os comandos de gestão de runs funcionam de dentro do projeto (como o `git`), enquanto `train` e `check` precisam de um alvo explícito para evitar treinar o projeto errado por acidente.

**`verify` guarda o run de verificação no store do projeto original, não em `code/`.**
O `verify` treina a partir de `code/` (snapshot do run original), mas armazenar o run em `code/.mlt/` tornaria a verificação invisível no `mlt runs list` do projeto. A solução foi adicionar o parâmetro `_store_project_dir` ao `trainer.train()`: o RunStore é inicializado com o projeto original, e o run de verificação aparece na listagem com o marcador `[V]` e com `verification_of` preenchido no manifesto. O parâmetro é prefixado com `_` para sinalizar uso interno.

**`verify` recusa se o hash dos dados mudou — saída 8.**
Sem `--snapshot-data`, a reprodutibilidade dos dados é de identidade. Se o hash atual diverge do manifesto original, o `verify` recusa com `OperationRefusedError` (exit 8) e orienta o usuário a usar `mlt diff` ou `--snapshot-data`. Alternativa seria treinar assim mesmo e avisar — mas isso produziria um run com dados diferentes, o que contradiz o objetivo do verify.

**`iris.csv` embutido no template iris.**
O template iris não tinha dado de exemplo, o que quebrava a promessa do `mlt init` ("próximos passos: cd, check, train") — o cientista precisava gerar o CSV manualmente antes de poder treinar. A decisão foi incluir `data/iris.csv` (150 linhas, ~6 KB, domínio público) diretamente em `src/mlt/templates/iris/data/`. O `mlt init` já copia tudo via `template_dir.rglob("*")`, sem nenhuma mudança de código. Alternativa avaliada: um script `generate_data.py` no template, mais explícito mas com mais atrito. O CSV embutido foi preferido pela simplicidade.

**`--rebuild` adiado.**
O `--rebuild` força a reconstrução do ambiente cacheado no venv. A implementação exigiria mudanças no `VenvRunner` (deletar o diretório do env e recriar) além da flag na CLI. Como é um caso de borda operacional (ambiente corrompido) que não afeta a correção, foi adiado. O `--force` cobre o caso mais comum (retreinar ignorando a idempotência).

## Ambiguidades e interpretações

Quando o enunciado deixou algo aberto, escolhi a interpretação mais simples que mantém a ferramenta útil de ponta a ponta.

| Ponto | Interpretação escolhida | Motivo |
| --- | --- | --- |
| Formato da ferramenta | Pacote Python com CLI, usando Docker como isolamento | O usuário fornece só Python; o resto é da ferramenta |
| Contrato mínimo | Classe `MLModel` com `fit`, `predict`, `save` e `load`; `SklearnModel` opcional | Convenção única e verificável |
| Isolamento | Docker como principal, venv como alternativa | Não depende de Docker na máquina de quem avalia |
| API de inferência | Um serviço por run, sem autenticação | Abstrai o backend; o resto é evolução |
| Artefatos e métricas associados à execução | Pasta do run com manifesto, e índice SQLite | Rastreável sem serviço externo |
| Dataset público e modelo simples | Iris com RandomForest | O foco é a ferramenta |
| Arquitetura em produção | Desenho AWS, não implementado | É opcional no enunciado |

**"Somente código Python":** interpretei como "sem infraestrutura nem servidor", e não como "sem nenhuma convenção". O enunciado pede que eu defina o contrato mínimo e o justifique, então alguma estrutura é esperada. O obrigatório é `requirements.txt` e `model.py`; `params.yaml` e `schema.yaml` são opcionais. Eles não são Python porque configuração lida sem executar código é mais fácil de revisar, versionar e comparar entre runs.

**Atrito de adoção:** herdar de `MLModel` acopla o `model.py` à ferramenta, e um script existente precisa ser adaptado. Mitigações: o `mlt init` entrega o esqueleto, a classe fica em um subpacote sem dependências e continua testável fora da ferramenta.

**Guia do cientista:** o contrato precisa estar visível para quem usa a ferramenta, e não só para quem a projeta. O README traz um guia de uma página (classe base a escolher, o que `fit` e `predict` recebem, `requirements.txt` completo, dados e schema), e o `init` oferece os templates `iris` (scikit-learn) e `blank` (esqueleto de `MLModel`). Os dados ficam em `<projeto>/data/` por padrão, ou em outra pasta indicada por `--data`. Erros dentro do `predict` só aparecem ao servir, porque o `check` não treina; isso está dito no guia.

**Validação do contrato em duas etapas:** o `check` é estático e não importa o `model.py`, porque as dependências do cientista só existem no ambiente isolado, e não no host. A validação profunda (importar, instanciar e conferir os métodos) roda dentro do ambiente, sem rede, antes de criar o run. Assim, erro de contrato continua sem criar run.

**`save` e `load` sobre um diretório local:** o modelo escreve e lê um diretório criado pela ferramenta. Quem transporta o pacote entre o disco e um registry remoto é o `RunStore`, então o modelo nunca fala com o registry e o contrato do cientista não muda quando o `RunStore` troca. O limite é que modelos muito grandes ou que gravam direto em destino remoto pediriam uma URI, o que cabe na regra de injeção de parâmetros por nome.

## Alternativas consideradas

Cada alternativa abaixo foi avaliada e descartada, com o motivo.

| Alternativa | Por que foi descartada |
| --- | --- |
| Contrato por função (`train()`) | Espalharia convenções; a classe dá reuso e um ponto de verificação |
| Schema como atributo Pydantic na classe | Mantém a definição no código; o YAML permite revisar o que foi inferido |
| MLflow para registro de runs | Pesado para o escopo; a interface `RunStore` permite adotá-lo depois |
| Dados embutidos na imagem | Imagem pesada e sem cache; o volume somente leitura resolve |
| CLI como cliente de um serviço remoto desde o início | É uma plataforma, e o enunciado não a pede; o núcleo separado permite a evolução |
| `.mlt/` no diretório atual | O resultado dependeria de onde o comando é executado; ancorei ao projeto |
| Runtime isolado em dois ambientes | Dois processos e um protocolo; adiado, com `MLModel` já em subpacote sem dependências |
| Seed só em `params.yaml` | Dependeria de o cientista declará-la; flag e variável de ambiente garantem o registro |
| Estatísticas de dados no manifesto | Só identificam, não reproduzem; fora do MVP |

## Fora do escopo e evolução

O MVP entrega o fluxo local completo; o que segue ficou de fora por escolha.

**Fora do escopo:** autenticação, versionamento de dados, rollback e A/B, monitoramento e drift, GPU, treino distribuído, UI, limites de CPU e memória, limpeza de runs antigos e versionamento de rota da API.

**Evolução, na ordem em que eu faria:**

1. Isolar o runtime do `mlt` do ambiente do cientista (SDK sem dependências mais um processo trabalhador).
2. `DataSource` remoto (S3 e lakehouse, com snapshot).
3. CLI como cliente de uma API de controle, com `Runner` em AWS Batch ou SageMaker Training.
4. `RunStore` em S3 e DynamoDB e, depois, um registry de runs e modelos com aliases e promoção.
5. Bloco de saída opcional no `schema.yaml` e suporte a `predict_proba`.

**Também como evolução:** reduzir o atrito de adoção, aceitando um estimador do scikit-learn ou uma função de treino simples por meio de um adaptador, sem exigir a herança de `MLModel`.

**Suporte a PySpark:** uma `SparkModel` com `save` e `load` nativos, `fit` com injeção de parâmetros por nome, `Runner` de cluster e `DataSource` por URI.

**Prioridades de entrega (7 dias):** P0 é o que não pode faltar: contrato, treino em Docker e venv, manifesto e RunStore, `runs list/show`, `serve` em primeiro plano com `/predict`, testes, README e documentos. P1: `check`, `init` com template `iris`, `verify`, `diff`. P2, cortado primeiro se faltar tempo: `--snapshot-data`, `serve --detach`, `ps`, `stop`, `logs`, reconciliação de runs interrompidos, template `blank`, `--json` e `runs list --status`. O `serve --detach` é P2 porque, sem `stop`, um serviço em segundo plano não teria como ser encerrado. Os fluxos de todos os itens P2 estão fechados no `design.md`, mesmo que não sejam implementados, e o que for cortado será listado aqui.

## Tempo gasto e o que faria diferente

Esta seção é do autor e fica para depois da implementação.

- Tempo aproximado gasto: a preencher.
- O que faria diferente com mais tempo: a preencher.
