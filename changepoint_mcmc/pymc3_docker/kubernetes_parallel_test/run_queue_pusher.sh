kubectl run -i --tty redis --overrides='
{
  "apiVersion": "batch/v1",
  "spec": {
    "template": {
      "spec": {
        "containers": [
          {
            "name": "temp",
            "image": "redis",
            "args": [
              "/bin/sh"
            ],
            "stdin": true,
            "stdinOnce": true,
            "tty": true,
            "volumeMounts": [{
              "mountPath": "/media/bigdata/firing_space_plot/changepoint_mcmc/pymc3_docker/kubernetes_parallel_test",
              "name": "store"
            }]
          }
        ],
        "volumes": [{
          "name":"store",
          "emptyDir":{}
        }]
      }
    }
  }
}
'  --image=redis --command "/bin/sh"
