import grpc
from concurrent import futures
import time
import os
import sys
import traceback

from generated.health.v1 import health_pb2_grpc, health_pb2
from generated.routegenerator.v1 import routegenerator_pb2_grpc, routegenerator_pb2

from algo.generate import generate
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)

class HealthServicer(health_pb2_grpc.HealthServicer):
    def Check(self, request, context):
        print(f"Health check requested for service: {request.service}")
        return health_pb2.HealthCheckResponse(
            status=health_pb2.HealthCheckResponse.SERVING
        )

class RouteGeneratorServicer(routegenerator_pb2_grpc.RouteGeneratorServicer):
    def GenerateRoutes(self, request, context):
        start_time_server = time.time()
        print(f"\n>>> Received GenerateRoutes request (num_routes={request.num_routes})")
        response = routegenerator_pb2.GenerateRoutesResponse()

        try:
            start_lon_lat = list(request.start_point_lon_lat) if request.start_point_lon_lat else None
            end_lon_lat = list(request.end_point_lon_lat) if request.end_point_lon_lat else None

            algo_result = generate(
                request.geojson_geometry,
                request.num_routes,
                start_lon_lat=start_lon_lat,
                end_lon_lat=end_lon_lat
            )

            response.visualization_png = algo_result.get("visualization_png", '')
            response.routes_geojson = algo_result.get("routes_geojson", '{"type": "FeatureCollection", "features": []}')
            response.route_costs.extend(algo_result.get("route_costs", []))
            response.status_message = algo_result.get("status_message", "Статус не определен")

            if not algo_result.get("success", False):
                 context.set_code(grpc.StatusCode.INTERNAL)
                 context.set_details(response.status_message)
                 print(f"<<< Request failed in algo: {response.status_message}")
            else:
                 print(f"<<< Request processed successfully: {response.status_message}")


        except Exception as e:
            print(f"[SERVER ERROR] Unhandled exception processing GenerateRoutes: {e}")
            traceback.print_exc()
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Внутренняя ошибка сервера: {e}")
            response.status_message = f"Внутренняя ошибка сервера: {e}"
            response.visualization_png = ''
            response.routes_geojson = '{"type": "FeatureCollection", "features": []}'
            del response.route_costs[:]

        print(f">>> Request handled in {time.time() - start_time_server:.2f} seconds")
        return response

class FieldAssistantServicer(routegenerator_pb2_grpc.FieldAssistantServicer):
    def AskAssistant(self, request, context):
        print(f">>> FieldAssistant called with: {request.question}")

        user_prompt = f"""\
        Ты — полевой помощник для водителя автозимника. Водитель сейчас на координатах {request.current_location_lon_lat}, \
        находится на маршруте {request.route_id}. Вот его вопрос: "{request.question}"

        Ответь кратко и по делу, если ситуация опасная — предупреди, и посоветуй действия.
        """

        try:
            inputs = tokenizer(user_prompt, return_tensors="pt").to("cuda")
            outputs = model.generate(**inputs, max_new_tokens=256)
            answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

            return routegenerator_pb2.AssistantResponse(
                answer=answer,
                recommended_actions=["Оценить толщину льда", "Остановиться и позвонить в диспетчерскую"],
                warning_level="danger" if "опасн" in answer.lower() or "лед" in answer.lower() else "warning"
            )
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return routegenerator_pb2.AssistantResponse(answer="Ошибка обработки запроса", warning_level="warning")

def serve():
    max_message_length = 10 * 1024 * 1024
    server = grpc.server(
             futures.ThreadPoolExecutor(max_workers=int(os.getenv("GRPC_MAX_WORKERS", "4"))),
             options=[
                 ('grpc.max_send_message_length', max_message_length),
                 ('grpc.max_receive_message_length', max_message_length)
             ]
         )

    health_pb2_grpc.add_HealthServicer_to_server(HealthServicer(), server)
    routegenerator_pb2_grpc.add_RouteGeneratorServicer_to_server(RouteGeneratorServicer(), server)
    routegenerator_pb2_grpc.add_FieldAssistantServicer_to_server(FieldAssistantServicer(), server)

    port = os.getenv("GRPC_PORT", "9090")
    listen_address = f'[::]:{port}'

    server.add_insecure_port(listen_address)

    print(f"gRPC Server starting on {listen_address}...")
    server.start()
    print("Server started.")
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("Stopping server...")
        server.stop(0)
        print("Server stopped.")

if __name__ == '__main__':
    serve()