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