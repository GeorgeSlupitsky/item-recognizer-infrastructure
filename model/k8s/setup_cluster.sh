#!/bin/bash

set -e 

echo "=== 0. Cleaning up any existing cluster ==="
# Вбиваємо будь-які існуючі переадресації портів
pkill -f "kubectl port-forward.*raycluster-kuberay-head-svc" || true
pkill -f "kubectl port-forward.*prometheus" || true
pkill -f "kubectl port-forward.*grafana" || true

# Видаляємо існуючий кластер, якщо він існує
kind delete cluster --name ray-cluster || true

echo "=== 1. Starting Kind cluster ==="
mkdir -p /tmp/kubeflow-data
kind create cluster --config kind/kind-config.yaml

# Очікуємо готовності кластера
echo "Waiting for cluster to be ready..."
kubectl wait --for=condition=ready nodes --all --timeout=300s

echo "=== 2. Installing Prometheus Stack ==="
# https://docs.ray.io/en/latest/cluster/kubernetes/k8s-ecosystem/prometheus-grafana.html
# Додаємо Helm репозиторії
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm repo update

# Створюємо namespace для prometheus
kubectl create namespace prometheus-system || true

# Встановлюємо kube-prometheus-stack
echo "Installing kube-prometheus-stack..."
helm install prometheus prometheus-community/kube-prometheus-stack \
    --namespace prometheus-system \
    --version 61.7.2 \
    -f monitoring/prometheus/prometheus-values.yaml

echo "=== 3. Waiting for Prometheus stack to be ready ==="
echo "Waiting for Prometheus operator..."
kubectl wait --for=condition=available --timeout=300s deployment/prometheus-kube-prometheus-operator -n prometheus-system

echo "Waiting for Prometheus StatefulSet to be created..."
while true; do
    if kubectl get statefulset prometheus-prometheus-kube-prometheus-prometheus -n prometheus-system >/dev/null 2>&1; then
        echo "✅ Prometheus StatefulSet found"
        break
    fi
    echo "⏳ Prometheus StatefulSet not found yet, waiting..."
    sleep 10
done

echo "Waiting for Prometheus pod to be ready..."
kubectl wait --for=condition=ready --timeout=300s pod/prometheus-prometheus-kube-prometheus-prometheus-0 -n prometheus-system

echo "Waiting for Grafana..."
kubectl wait --for=condition=ready --timeout=300s pod -l app.kubernetes.io/name=grafana -n prometheus-system

echo "=== 4. Installing KubeRay operator ==="
helm install kuberay-operator kuberay/kuberay-operator --version 1.3.2 

echo "=== 5. Waiting for KubeRay operator to be ready ==="
kubectl wait --for=condition=available --timeout=300s deployment/kuberay-operator

echo "Checking operator status..."
kubectl get deployment kuberay-operator -o wide

echo "=== 6. Installing Ray cluster with SCALE-TO-ZERO autoscaling ==="
helm install raycluster kuberay/ray-cluster --version 1.3.2 -f ray-cluster/ray-cluster-values.yaml

echo "=== 7. Installing monitoring configuration ==="
kubectl apply -f monitoring/prometheus/ray-servicemonitor.yaml
kubectl apply -f monitoring/prometheus/ray-podmonitor.yaml
kubectl apply -f monitoring/prometheus/ray-prometheus-rules.yaml

echo "=== 8. Waiting for Ray cluster to be ready ==="

wait_for_resource_to_exist() {
    local resource_type=$1
    local selector=$2
    local description=$3
    
    echo "Waiting for $description to be created..."
    while true; do
        if kubectl get $resource_type -l $selector --no-headers 2>/dev/null | grep -q .; then
            echo "✅ $description found"
            break
        fi
        echo "⏳ $description not found yet, waiting..."
        sleep 10
    done
}

wait_for_resource_to_exist "pod" "ray.io/node-type=head" "head pod"

echo "Waiting for head pod to be ready..."
kubectl wait --for=condition=ready --timeout=600s pod -l ray.io/node-type=head

echo "=== 9. Verifying cluster health ==="
# Перевіряємо статус кластера
echo "Checking cluster status..."
kubectl get pods -l ray.io/cluster-name -o wide

# Перевіряємо статус Ray всередині кластера
echo "Checking Ray status..."
HEAD_POD=$(kubectl get pod -l ray.io/node-type=head -o jsonpath='{.items[0].metadata.name}')
if [ ! -z "$HEAD_POD" ]; then
    echo "Head pod found: $HEAD_POD"
    
    echo "Waiting for Ray to be ready inside head pod..."
    for i in {1..10}; do
        if kubectl exec $HEAD_POD -- ray status 2>/dev/null; then
            echo "✅ Ray is ready!"
            break
        fi
        echo "⏳ Ray not ready yet, attempt $i/10..."
        sleep 15
    done
    
    # Тепер, коли Ray готовий, спробуємо отримати dashboard
    echo "Extracting Ray dashboards from ready Ray cluster..."
    RAY_SESSION=$(kubectl exec $HEAD_POD -c ray-head -- find /tmp/ray -name "session_*" -type d 2>/dev/null | head -1)
    if [ ! -z "$RAY_SESSION" ]; then
        echo "✅ Found Ray session: $RAY_SESSION"
        
        # Спочатку налаштовуємо port-forward для Grafana
        echo "Setting up Grafana port-forward..."
        pkill -f "kubectl port-forward.*grafana" || true
        kubectl port-forward svc/prometheus-grafana -n prometheus-system 3000:80 > /dev/null 2>&1 &
        sleep 5
        
        # Чекаємо готовності Grafana з кількома спробами
        echo "Waiting for Grafana to be ready..."
        GRAFANA_READY=false
        for i in {1..12}; do
            if curl -s --connect-timeout 5 http://localhost:3000/api/health >/dev/null 2>&1; then
                echo "✅ Grafana is ready!"
                GRAFANA_READY=true
                break
            fi
            echo "⏳ Grafana not ready yet, attempt $i/12..."
            sleep 10
        done
        
        if [ "$GRAFANA_READY" = true ]; then
            # Імпортуємо всі доступні Ray дашборди
            echo "Importing all Ray dashboards..."
            DASHBOARDS_IMPORTED=0
            
            # Отримуємо список всіх дашбордів
            DASHBOARD_FILES=$(kubectl exec $HEAD_POD -c ray-head -- find $RAY_SESSION/metrics/grafana/dashboards/ -name "*.json" -type f 2>/dev/null || echo "")
            
            if [ ! -z "$DASHBOARD_FILES" ]; then
                for DASHBOARD_FILE in $DASHBOARD_FILES; do
                    DASHBOARD_NAME=$(basename $DASHBOARD_FILE .json)
                    echo "📊 Processing dashboard: $DASHBOARD_NAME"
                    
                    # Копіюємо dashboard
                    if kubectl cp $HEAD_POD:$DASHBOARD_FILE monitoring/grafana/${DASHBOARD_NAME}.json -c ray-head 2>/dev/null; then
                        # Перевіряємо JSON
                        if jq empty monitoring/grafana/${DASHBOARD_NAME}.json 2>/dev/null; then
                            # Імпортуємо через API
                            DASHBOARD_JSON=$(cat monitoring/grafana/${DASHBOARD_NAME}.json)
                            API_RESPONSE=$(curl -s -X POST http://localhost:3000/api/dashboards/db \
                                -u "admin:prom-operator" \
                                -H "Content-Type: application/json" \
                                -d "{
                                    \"dashboard\": $DASHBOARD_JSON,
                                    \"overwrite\": true,
                                    \"message\": \"Ray $DASHBOARD_NAME imported from live cluster\"
                                }" 2>/dev/null)
                            
                            if echo "$API_RESPONSE" | jq -e '.status == "success"' >/dev/null 2>&1; then
                                DASHBOARD_URL=$(echo "$API_RESPONSE" | jq -r '.url')
                                echo "  ✅ $DASHBOARD_NAME imported successfully!"
                                echo "  📊 URL: http://localhost:3000$DASHBOARD_URL"
                                DASHBOARDS_IMPORTED=$((DASHBOARDS_IMPORTED + 1))
                            else
                                echo "  ❌ Failed to import $DASHBOARD_NAME"
                            fi
                        else
                            echo "  ⚠️ Invalid JSON for $DASHBOARD_NAME"
                        fi
                    else
                        echo "  ⚠️ Failed to copy $DASHBOARD_NAME"
                    fi
                done
                
                echo "🎉 Successfully imported $DASHBOARDS_IMPORTED Ray dashboards!"
            else
                echo "⚠️ No dashboard files found"
            fi
        else
            echo "⚠️ Grafana not ready after 2 minutes, dashboard import skipped"
        fi
    else
        echo "⚠️ Ray session not found in cluster"
    fi
else
    echo "❌ Head pod not found"
    exit 1
fi

echo "=== 10. Setting up port forwarding ==="
# Вбиваємо будь-які існуючі переадресації портів
pkill -f "kubectl port-forward.*raycluster-kuberay-head-svc" || true
pkill -f "kubectl port-forward.*prometheus-kube-prometheus-prometheus" || true

# Налаштовуємо переадресації портів у фоновому режимі
echo "Starting port forwards..."

# Ray services
kubectl port-forward service/raycluster-kuberay-head-svc 8265:8265 > /dev/null 2>&1 &
kubectl port-forward service/raycluster-kuberay-head-svc 10001:10001 > /dev/null 2>&1 &
kubectl port-forward service/raycluster-kuberay-head-svc 8000:8000 > /dev/null 2>&1 &

# Prometheus (Grafana port-forward вже налаштований раніше)
kubectl port-forward svc/prometheus-kube-prometheus-prometheus -n prometheus-system 9090:9090 > /dev/null 2>&1 &

# Чекаємо встановлення переадресацій портів
echo "Waiting for port forwards to establish..."
sleep 15

# Перевірка доступності
echo "Checking service availability..."
curl -s --connect-timeout 3 http://localhost:8265 > /dev/null && echo "✅ Ray Dashboard accessible" || echo "⚠️  Ray Dashboard may need more time to start"
curl -s --connect-timeout 3 http://localhost:9090 > /dev/null && echo "✅ Prometheus accessible" || echo "⚠️  Prometheus may need more time to start"
curl -s --connect-timeout 3 http://localhost:3000 > /dev/null && echo "✅ Grafana accessible" || echo "⚠️  Grafana may need more time to start"

echo "=== 11. Verifying metrics collection ==="
echo "Checking if Ray metrics are being collected..."
sleep 5

# Перевіряємо, чи Ray експортує метрики
RAY_METRICS_URL="http://localhost:8265/api/prometheus_metrics"
if curl -s --connect-timeout 5 "$RAY_METRICS_URL" | head -10 > /dev/null 2>&1; then
    echo "✅ Ray metrics endpoint is responding"
else
    echo "⚠️  Ray metrics endpoint may need more time to start"
fi

echo "=== Ray cluster with Prometheus and Grafana setup completed! ==="
echo ""
echo "🎉 Services available at:"
echo "   📊 Ray Dashboard:     http://localhost:8265"
echo "   📈 Prometheus:        http://localhost:9090"  
echo "   📊 Grafana:           http://localhost:3000"
echo "      Grafana login:     admin / prom-operator"
