from io import BytesIO

from locust import HttpUser, task, between
from PIL import Image


class UserBehavior(HttpUser):
    wait_time = between(1, 3)

    @task
    def health_check(self):
        self.client.get("/health")

    @task
    def ready_check(self):
        self.client.get("/ready")

    @task
    def v1_predict(self):
        image = Image.new(
            mode="RGB",
            size=(516, 516),
            color=(255, 0, 0),
        )
    
        input_image = BytesIO()
        image.save(input_image, format="PNG")

        input_image.seek(0)

        with self.client.post(
            "/v1/predict",
            files={
                "file": (
                    "test_name.png",
                    input_image,
                    "image/png",
                ),
            },
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f'Request failed with status code {response.status_code}')