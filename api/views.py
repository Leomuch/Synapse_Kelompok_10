import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from api.model.inference import Model

model = Model.from_path("api/model/garden/model.joblib")


@csrf_exempt
def predict(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body.decode("utf-8"))
            user_data = data.get("data")

            if user_data is None:
                return JsonResponse({"error": "No data provided"}, status=400)

            prediction = model.predict_from_data(user_data)

            return JsonResponse(
                {"message": "Data received", "prediction": prediction}, status=200
            )
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON format"}, status=400)
    else:
        return JsonResponse(
            {"error": "This endpoint only supports POST requests."}, status=405
        )