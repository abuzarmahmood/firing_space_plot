helm upgrade --cleanup-on-fail \
  --install juphub jupyterhub/jupyterhub \
  --namespace juphub \
  --create-namespace \
  --version=1.2.0 \
  --values config.yaml
