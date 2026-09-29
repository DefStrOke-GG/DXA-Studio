# Third party notices

DXA Studio использует сторонние библиотеки. Точные версии прямых зависимостей зафиксированы в файлах `requirements.txt`; транзитивные зависимости устанавливаются их менеджерами пакетов.

| Компонент | Лицензия по метаданным установленного пакета | Проект |
|---|---|---|
| PyTorch | BSD-3-Clause | <https://pytorch.org/> |
| torchvision | BSD | <https://github.com/pytorch/vision> |
| FastAPI | MIT | <https://github.com/fastapi/fastapi> |
| Uvicorn | BSD-3-Clause | <https://www.uvicorn.org/> |
| python-multipart | Apache-2.0 | <https://github.com/Kludex/python-multipart> |
| HTTPX | BSD-3-Clause | <https://github.com/encode/httpx> |
| Requests | Apache-2.0 | <https://requests.readthedocs.io/> |
| pydicom | MIT | <https://github.com/pydicom/pydicom> |
| NumPy | BSD-3-Clause | <https://numpy.org/> |
| SciPy | BSD-3-Clause | <https://scipy.org/> |
| scikit-learn | BSD-3-Clause | <https://scikit-learn.org/> |
| pandas | BSD-3-Clause | <https://pandas.pydata.org/> |
| Pillow | HPND | <https://python-pillow.org/> |
| openpyxl | MIT | <https://openpyxl.readthedocs.io/> |
| Matplotlib | PSF-based | <https://matplotlib.org/> |
| Streamlit | Apache-2.0 | <https://streamlit.io/> |
| pylibjpeg | MIT | <https://github.com/pydicom/pylibjpeg> |
| pylibjpeg-libjpeg | GPL-3.0 | <https://github.com/pydicom/pylibjpeg-libjpeg> |
| pytest | MIT | <https://pytest.org/> |

Перед распространением готового контейнера необходимо проверить полные тексты лицензий всех установленных пакетов, включая транзитивные и бинарные зависимости. Этот файл — сводка, а не замена оригинальных лицензий.

## Модели

Код использует архитектуры и начальные веса ImageNet из torchvision. Финальные веса DXA Studio обучены командой и в публичный репозиторий не входят. Их источник, версия и контрольные суммы описываются в [`MODEL_CARD.md`](MODEL_CARD.md) и `model_manifest.json` внутри model bundle.

## Лицензия проекта

Отдельная лицензия на исходный код DXA Studio владельцем репозитория пока не выбрана. Публичная доступность кода не означает автоматическую передачу прав на использование или распространение.
