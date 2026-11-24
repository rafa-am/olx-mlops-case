# AI_USAGE.md: Case MLOps

## Ferramentas usadas

Usei o Claude em conversa para planejar o design antes de escrever código, e quadros brancos colaborativos para registrar cada decisão aprovada.

- **Claude (chat):** levantar alternativas e custos, detalhar contratos, redigir propostas.
- **Quadros brancos:** manter o desenho vivo (fluxo, contratos, dados, manifesto, API, testes).
- **Implementação:** Claude Code (CLI) gerou o código da etapa 1 sob revisão do autor.

## Como a IA entrou no processo

O design foi feito em conversa iterativa, ponto a ponto, antes de qualquer implementação.

Cada ponto seguiu o mesmo ciclo:

1. A IA propunha uma solução com alternativas e custos.
2. Eu questionava, escolhia ou ajustava.
3. A decisão era registrada no quadro e, depois, nos documentos.

A ordem foi: leitura do enunciado, plano inicial, contrato do cientista, dados e rastreabilidade, manifesto, API e serving, demo, seed e falhas, estrutura do repositório, testes e ambiente.

## Fluxos e prompts que funcionaram

O que mais rendeu foi pedir alternativas com custos antes de decidir, e fazer perguntas diretas sobre garantias.

- **Perguntas sobre limites:** "No MVP, como eu tenho garantia de reprodutibilidade de uma run se os dados baixados mudam?" levou à separação entre rastreabilidade de identidade e de conteúdo.
- **Perguntas de escopo:** "Isto talvez aumente a complexidade; vale abordar assim ou deixar como evolução?" produziu a decisão de separar o núcleo da CLI e adiar o serviço.
- **Revisão de lacunas:** "O que precisamos definir ainda? Há contratos em aberto?" listou os itens que fecharam o design.
- **Registro contínuo:** cada decisão aprovada era anotada no quadro antes de seguir, o que evitou perder o fio.

## O que a IA gerou e foi corrigido

Várias propostas da IA foram mudadas por eu ter questionado, e cada mudança melhorou o design.

| A IA propôs | O que decidi | Motivo |
| --- | --- | --- |
| Contrato por função `train()` | Contrato por classe `MLModel` | Organização e reuso de código |
| Schema como atributo Pydantic na classe | Schema em `schema.yaml`, declarado ou inferido | Tira a definição do código e permite revisar o inferido |
| Pasta `.mlt/` no diretório atual | `.mlt/` ancorada ao projeto | O resultado dependia de onde o comando era executado |
| Seed só pela flag da CLI | Flag e variável de ambiente, com origem registrada | Cobre scripts e CI, sem perder a rastreabilidade |
| Reprodutibilidade apresentada como garantia | Dividida em identificar, verificar e congelar | Sem congelar só há rastreabilidade de identidade |
| "Predições do mesmo tamanho que as entradas" | Distinção entre o número de itens e a forma de cada predição | A frase era imprecisa e mascarava saídas de outra dimensão |
| Estatísticas de dados no MVP | Nota fora do MVP | Identificam a mudança, mas não reproduzem o conteúdo |
| Run sem o código e o params.yaml (inconsistência entre design e documentos) | Run autocontido, com code/ e params.yaml; o verify retreina a partir dele | Achado na revisão final; sem isso o run não reproduz se a raiz mudar |
| Escopo completo para 7 dias | Prioridades P0, P1 e P2, cortando P2 primeiro | O enunciado não pede uma plataforma completa e avalia a priorização |
| Venv usando o "Python do host" | Python que executa o mlt (sys.executable), sem procurar outras versões | Perguntei como isso era garantido; o termo era ambíguo e o resultado dependeria do PATH |
| Versão do Python só pela flag --python | Arquivo opcional .python-version, com precedência flag, arquivo, padrão | O requirements.txt não declara o Python; o projeto precisa carregar essa informação |
| check exigindo a pasta data/ | check consulta o DataSource; o treino materializa os dados em diretório local | Perguntei se o snapshot precisa estar local; com lakehouse não precisaria |
| verify completo (tolerância ajustável, hash do artefato) | Versão mínima, cortável; o mesmo argumento fica num teste | Questionei a utilidade real; o valor está na demonstração de reprodutibilidade |
| serve --detach como P0 | serve --detach como P2, junto com ps, stop e logs | Ao fechar os fluxos P2, apareceu a dependência: sem stop não há como encerrar o serviço |

Também corrigi erros de layout da IA nos quadros (texto transbordando as seções) e pedi a separação em dois quadros quando o primeiro ficou grande demais.

**Contrato em camadas:** a IA sugeriu um `predict` padrão no núcleo, baseado em Pipeline do scikit-learn, para reduzir o atrito. Questionei o acoplamento a um tipo de artefato e a dificuldade de suportar outros, como PySpark. Decidimos por um núcleo agnóstico com `SklearnModel` como conveniência opcional.

## Como verifiquei o que a IA produziu

No design, nenhuma proposta virou decisão sem ser questionada (veja a tabela acima).

Na implementação, a verificação é feita com os testes dos fluxos críticos e com a execução do exemplo ponta a ponta.

### Etapa 1 — Esqueleto, MLModel, contrato e CLI (2026-11-24)

**O que a IA gerou:** pyproject.toml, src/mlt/{core/errors.py, core/contract.py, model/base.py, model/sklearn.py, cli/main.py} e os quatro arquivos de teste (37 testes).

**Ajuste feito durante a implementação:** o teste `test_no_args_shows_help` falhou com exit code 2 porque Click retorna 2 para grupo sem subcomando. A IA corrigiu adicionando `invoke_without_command=True` e um callback que exibe o help; nenhuma mudança de contrato.

**Como o autor verificou:**
- Rodou `pytest` até 37/37 passarem.
- Executou `mlt --version` manualmente e conferiu os três campos (versão, python, docker).
- Leu os arquivos gerados e conferiu que `MLModel` não tem imports externos e que os imports de joblib/pandas em `SklearnModel` são lazy.

**Tempo gasto:** ~25 min (geração + 1 rodada de fix do teste).

### Etapa 2 — Treino em venv, DataSource, RunStore, fingerprint, seed (2026-11-24)

**O que a IA gerou:** seed.py, fingerprint.py, datasources/{base,local}.py, runstore/{manifest,store}.py, runners/{base,venv}.py, _worker/train.py, core/trainer.py, templates/iris/, conftest.py e 5 arquivos de teste (57 novos testes, 10 de integração).

**Nenhum ajuste foi necessário:** todos os 94 testes passaram na primeira execução completa.

**Como o autor verificou:**
- Rodou `pytest` até 94/94 passarem (47 unitários + 10 integração + 37 de etapa 1).
- Conferiu o manifesto JSON gerado em `.mlt/runs/<run_id>/manifest.json` após treino real com o projeto iris.
- Leu o `train.log` do run para confirmar que stdout/stderr do worker chegam corretamente.
- Revisou o `_worker/train.py` e o `trainer.py` para confirmar que o run é deletado (não marcado failed) quando o worker sai com código 3.

**Tempo gasto:** ~15 min de geração + 8:36 de execução dos testes de integração (primeira vez, download de pacotes para o venv).

## Reflexões

Rascunho do autor, a revisar ao fim da implementação.

- **Onde a IA ajudou mais:** levantar alternativas e custos, e listar o que ainda estava em aberto.
- **Onde exigiu mais cuidado:** afirmações de garantia (reprodutibilidade, rastreabilidade) e detalhes de formato, que precisei questionar para ficarem precisos.
