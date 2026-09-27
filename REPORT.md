## 2

### 2.1

ссылка на успешный прогон: https://github.com/DmitriyZhariy/asl_classifier/commit/baf21492dbf9f0e924b5ad5d7ca1b9eaab757e57

ссылка на страницу пакета: https://github.com/DmitriyZhariy/asl_classifier/pkgs/container/asl-service%2Fsha-baf21492dbf9f0e924b5ad5d7ca1b9eaab757e57

---

### 2.2 

ссылка на PR: https://github.com/DmitriyZhariy/asl_classifier/pull/4

---

### 2.3

поломка конфига: https://github.com/DmitriyZhariy/asl_classifier/actions/runs/36342646156
починка конфика: https://github.com/DmitriyZhariy/asl_classifier/actions/runs/36343240717

как понять причину ошибки: в deploy на степе service сломался. diagnostic говорит, что заданной модели в пути нет:
onnxruntime.capi.onnxruntime_pybind11_state.NoSuchFile: [ONNXRuntimeError] : 3 : NO_SUCHFILE : Load model from artifacts/shufflenet_v999.onnx failed:Load model artifacts/shufflenet_v999.onnx failed. File doesn't exist

---

поломка секрета: https://github.com/DmitriyZhariy/asl_classifier/actions/runs/36343566809
починка секрета: https://github.com/DmitriyZhariy/asl_classifier/actions/runs/36344206009

как понять причину ошибки: поломка в deploy на степе service. в диагностике говорится, что не получилось создать api сервис, проблема в том, что его не получилось собрать:
error: Internal error occurred: unable to upgrade connection: container not found ("api")

---

поломка ресурсов: https://github.com/DmitriyZhariy/asl_classifier/actions/runs/36344502532
починка ресурсов: https://github.com/DmitriyZhariy/asl_classifier/actions/runs/36345485498

как понять причину ошибки: Все три пода asl-service не могут получить узел does not have a host assigned, хотя Postgres на том же узле спокойно запускается, поэтому проблема в спецификации пода asl-service

---

## 3

1) первый прогон - 42с (https://github.com/DmitriyZhariy/asl_classifier/actions/runs/36123884593/job/108035558276), второй прогон - 34с (https://github.com/DmitriyZhariy/asl_classifier/actions/runs/36232086372/job/108377087095). Второй больше времени потратил на buildx (12с против 5с), но сильно сэкономил в push (13с против 28с). Во втором случае он подгружает неизмененные элементы (python:3.12-slim например)
2) эти поды появляются на степе service в kubectl apply -f k8s/ который создает временные поды, опираясь на deployment.yaml, и они могут не запуститься, из-за чего появляется ImagePullBackOff. Следующей командой kubectl set image deploy/asl-service api="$IMAGE" - запускает замену с рабочим образом, прошлый образ удаляется. И только после этого сервис проверяет работоспособность образа в kubectl rollout status deploy/asl-service --timeout 180s, поэтому ничего не падает.
3) github - ci.yml, deploy, get-secrets, secrets.DB_PASSWORD - postgres.yaml через secretKeyRef. Хранить пароль в configmap нельзя т.к. это открытый файл, шифровать его сложно и долго, он не предназначен для этого.
4) тесты падают с ошибкой - сервис не может корректно работать, и без needs: tests упаковываем поломанный сервис
5) в build мы поставили условие выполнения if: github.ref== 'refs/heads/main' (только в ветке main выполняется), а deploy выполняется только после build.
6) он нужен, чтобы запросы на создание базы данных не наложились друг на друга и не вызвали из-за этого ошибку. Имеется ввиду реплики базы данных.
7) ci(config) https://github.com/DmitriyZhariy/asl_classifier/actions/runs/36342646156/job/108685758254 (не смог выделить места для образов) -> ci(secrets) https://github.com/DmitriyZhariy/asl_classifier/actions/runs/36343566809/job/108688406330 (место уже выделено, но полностью собрать не получилось из-за поломанного секрета) -> ci(memory) https://github.com/DmitriyZhariy/asl_classifier/actions/runs/36344502532/job/108691054597 (образы собраны, код запущен, но падает из-за неправильного пути до модели)


---

## Работа с ошибками

к POST запросу добавил захват ошибок

```python
    ...
        ) as response:
            if response.status_code != 200:
                response.failure(f'Request failed with status code {response.status_code}')
```

Получил ошибку 422 - проблема в запросе. По итогу просто забыл сбросить указатель после обработки изображения input_image.seek(0). 

Отчет по нагрузке (решил взять нагрузку на predict, с ней можно больше придумать по оптимизации):

| User Count | RPS | median | p95 | max | Failure % |
| ----- | ----- | ----- | ----- | ----- | ----- |
| 10 | 1.801436840775051 | 20 | 29 | 43.279799981974065 | 0.00% |
| 50 | 8.28009469990297 | 18 | 36 | 79.40289995167404 | 0.00% |
| 100 | 16.164655389525237 | 20 | 49 | 140.00040001701564 | 0.00%|

- на всех тестах p95 отрывался от медианы
- при увеличении количества пользователей, RPS также растет, сервис до потолка не уперся
- ошибок нет

Хоть медиана и держится на ~20мс, правый хвост увеличивается достаточно быстро, при дальнейшем увеличении пользователей, задержка может стать ощутимой.

---

```bash
$ DATABASE_URL=postgresql://postgres:postgres@127.0.0.1:5432/asl uv run pytest
...
tests\test_api.py EEEEE [ 62%]
tests\test_integration.py E [ 75%]
tests\test_smoke.py EE [100%]
...
E               psycopg.OperationalError: connection failed: connection to server at "127.0.0.1", port 5432 failed: �����:  ������������ "postgres" �� ������ �������� ����������� (�� ������)

.venv\Lib\site-packages\psycopg\connection.py:126: OperationalError
...
```

Базу данных он видит, но подключится не может. Для начала просто пересоздам контейнеры и удалю volumes. Проблема осталась.

```bash

$ docker compose down -v
[+] down 3/3
 ✔ Container asl_classifier-db-1  Removed                                                                                                                                             0.4s
 ✔ Network asl_classifier_default Removed                                                                                                                                             0.4s
 ✔ Volume asl_classifier_pgdata   Removed                                                                                                                                             0.1s
(asl) 
Redmi@DmitriyZh MINGW64 /d/Study/asl_classifier (main)
$ docker compose up -d
[+] up 4/4
 ✔ Network asl_classifier_default Created                                                                                                                                             0.0s
 ✔ Volume asl_classifier_pgdata   Created                                                                                                                                             0.0s
 ✔ Container asl_classifier-db-1  Healthy                                                                                                                                             6.0s
 ✔ Container asl_classifier-api-1 Started                                                                                                                                             6.2s
(asl) 
Redmi@DmitriyZh MINGW64 /d/Study/asl_classifier (main)
$ DATABASE_URL=postgresql://postgres:postgres@127.0.0.1:5432/asl uv run pytest
================================================================================== test session starts ==================================================================================
platform win32 -- Python 3.12.12, pytest-9.1.1, pluggy-1.6.0
rootdir: D:\Study\asl_classifier
configfile: pyproject.toml
plugins: anyio-4.15.1, locust-2.46.6
collected 8 items                                                                                                                                                                        

tests\test_api.py EEEEE                                                                                                                                                            [ 62%]
tests\test_integration.py E                                                                                                                                                        [ 75%]
tests\test_smoke.py EE 
```

какие-то 2 процесса, которые слушают 5432

```bash
$ netstat -ano | findstr :5432
  TCP    0.0.0.0:5432           0.0.0.0:0              LISTENING       6736
  TCP    0.0.0.0:5432           0.0.0.0:0              LISTENING       26652
  TCP    [::]:5432              [::]:0                 LISTENING       26652
  TCP    [::]:5432              [::]:0                 LISTENING       6736
(asl) 
```

одна бд из докера сервиса, второй - сам postgresql что скачан локально

```bash
(asl) PS D:\Study\asl_classifier> Get-Process -Id 6736,26652 | Select-Object Id, ProcessName, Path

   Id ProcessName        Path                                                           
   -- -----------        ----                                                           
26652 com.docker.backend C:\Program Files\Docker\Docker\resources\com.docker.backend.exe
 6736 postgres 
```

6736 вырубил через диспетчер задач

```bash
$ netstat -ano | findstr :5432
  TCP    0.0.0.0:5432           0.0.0.0:0              LISTENING       26652
  TCP    [::]:5432              [::]:0                 LISTENING       26652
(asl) 
```

теперь все выполнилось:

```bash
$ DATABASE_URL=postgresql://postgres:postgres@127.0.0.1:5432/asl uv run pytest
================================================================================== test session starts ==================================================================================
platform win32 -- Python 3.12.12, pytest-9.1.1, pluggy-1.6.0
rootdir: D:\Study\asl_classifier
configfile: pyproject.toml
plugins: anyio-4.15.1, locust-2.46.6
collected 8 items                                                                                                                                                                        

tests\test_api.py .....                                                                                                                                                            [ 62%]
tests\test_integration.py .                                                                                                                                                        [ 75%]
tests\test_smoke.py ..                                                                                                                                                             [100%]
```

---

после добавления db в ci.yml, проверка вывела failed: Failed to initialize container postgres:16 - образ не смог впринципе собраться

по итогу, не вписал пробелы в env, из-за чего не мог считать конфиги.

---

после написания job build получаю ошибку:
Error: buildx failed with: ERROR: failed to build: invalid tag "ghcr.io/DmitriyZhariy/asl-service/sha-2184132a222155271504cd4c625e06c4c65cb481": repository name must be lowercase

В моем имени на github есть заглавные буквы, поменял команду сохранения IMAGE
```
      - run: |
          OWNER_LC=$(echo "${GITHUB_REPOSITORY_OWNER}" | tr '[:upper:]' '[:lower:]')
          echo "IMAGE=ghcr.io/${OWNER_LC}/asl-service/sha-${GITHUB_SHA}" >> "$GITHUB_ENV"
```

---

делаем отдельную ветку для работы с ci/cd

```powershell
(asl) PS D:\Study\asl_classifier> git switch -c 'ci-cd'      
Switched to a new branch 'ci-cd'
(asl) PS D:\Study\asl_classifier> git push -u origin ci-cd
Total 0 (delta 0), reused 0 (delta 0), pack-reused 0 (from 0)
remote: 
remote: Create a pull request for 'ci-cd' on GitHub by visiting:
remote:      https://github.com/DmitriyZhariy/asl_classifier/pull/new/ci-cd
remote: 
To https://github.com/DmitriyZhariy/asl_classifier.git
 * [new branch]      ci-cd -> ci-cd
branch 'ci-cd' set up to track 'origin/ci-cd'.
```

---

Добавил сохранение статус кода в базу данных. Также для этого добавил сохранение неуспешных вызовов predict. По какой-то причине, запросы (которые не 200) не сохраняются в базе данных

через обычный print в save_prediction понял, что при ошибке, db.save_prediction не запускается в принципе.

Проблема оказалась в bg.add_task - он добавляет задачу сохранения в бд на фон, пока тот ее выполняет, у нас запускается raise HTTPException и задача теряется.

Это скорее всего можно решить через asyncio, но пока эта проблема не в приоритете, поэтому для сохранения записей неудачных вызовов predict будем использовать синхронный вариант.


              request_id              |              ts               | model_version | prediction_class |                             all_probabilities                              |                                      input_metadata                                       | latency_ms | status_code 
--------------------------------------+-------------------------------+---------------+------------------+----------------------------------------------------------------------------+-------------------------------------------------------------------------------------------+------------+-------------
 b8d6d425-231f-4095-a738-24fff793a708 | 2026-09-26 08:10:03.378688+00 | 1.0.0         | A                | {"0": 0.511117696762085, "1": 0.178506538271904, "2": 0.31037577986717224} | {"filename": "ChatGPT Image 23 сент. 2026 г., 11_12_00.png", "content_type": "image/png"} |   66.79743 |         200
 0126280b-2003-4aaa-8967-ee63111afa1f | 2026-09-26 08:10:07.5797+00   | 1.0.0         |                  | null                                                                       | {"filename": "lecture02_2.pdf", "content_type": "application/pdf"}                        |   0.139938 |         415