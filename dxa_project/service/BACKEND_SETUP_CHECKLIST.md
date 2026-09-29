# Допоставка для backend

Код API в Git и `backend_handoff.zip` не содержит весов или исходных медицинских данных.

## Predict

Сначала определить корень клонированного проекта. В нём должны находиться `README.md`, каталог `dxa_project` и каталог `user_web_prototype`. Папка, в которой непосредственно лежат `augmentation`, `geometry_ml`, `service` и `tests`, — это уже `dxa_project`, то есть на один уровень ниже корня.

`dxa_project/outputs` отсутствует в чистом Git-репозитории намеренно: веса поставляются отдельно. Распаковать `backend_models_20260929.zip` в корень проекта, а не внутрь `dxa_project`, и сохранить пути из архива:

```text
dxa_project/outputs/final_20260929/
  protocol.json
  bundle/
    router.pt
    spine.pt
    hip.pt
    hip_mask.pt
    hip_points.pt
    artifact.pt
    scoliosis.pt
    spine_crests.pt
    training_config.json
    landmark_geometry_bounds.json
    model_manifest.json
    protocol.json
    evaluation/calibration.json
```

Проверить из корня проекта:

```powershell
Test-Path .\README.md
Test-Path .\user_web_prototype
Test-Path .\dxa_project\outputs\final_20260929\bundle\router.pt
```

Все три результата должны быть `True`. Если первые два пути не найдены, это не корень полного репозитория. Если не найден только `router.pt`, архив модели распакован не в тот каталог или ещё не распакован.

Из корня проекта, после установки зависимостей по API_GUIDE.md:

```powershell
python -m dxa_project.service --device auto --data-dir dxa_project/service_data
```

Проверить `GET http://127.0.0.1:8765/v1/health`: status=ready и active_model не пуст. Swagger: `/docs`. Автоматическая регистрация выполняется при отсутствии активной модели. Если веса лежат в другом месте, указать `--bootstrap-bundle <каталог bundle>`; соседний `protocol.json` обязателен.

## Partial fit

Дополнительно распаковать `backend_training_reference_20260929.zip` в тот же корень проекта. Это исходный датасет для фиксированной validation и проверки утечки: `Исследования`, `Размеченные/labels.csv`, geometry JSON и `dxa_project/outputs/manifest.csv`. Старые аугментации не требуются. Архив содержит медицинские изображения и разметку; передавать отдельно backend-разработчику.

В запросе нужен новый непустой набор human; model можно не передавать:
- `human`: снимки с подтверждённой/исправленной человеком разметкой;
- `model` (необязательно): остальные снимки с настоящей разметкой предсказания модели.

Frontend может автоматически собрать model из предсказанных, но не исправленных снимков. Исправленный человеком снимок должен заменить свою модельную версию, а human и дубликаты по пиксельным данным необходимо исключить из model. При пустом model запрос разрешён; сервис выполнит только human-этапы. Сервис сам не дополняет эти наборы.

Голый исходный DICOM без предсказания недостаточен для model. Передать DICOM из `originals/` результата predict с DXA_MODEL_V1 метаданными или явно передать geometry и targets из результата. human использует исправленную геометрию и метки. Источники из исходных validation/test и их пиксельные дубликаты запрещены для дообучения.

Пример запроса: `examples/partial_fit.json`; полный контракт и условные model-этапы: `API_GUIDE.md`.

Архивы моделей и reference-данных не включают новые экспериментальные веса. SHA256 и состав будут сохранены в `backend_resources_manifest.json`.
