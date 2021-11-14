Guide on how to run kubernetes work-queue parallel tutorial from:
https://kubernetes.io/docs/tasks/job/fine-parallel-processing-work-queue/

1) Download all files linked to in above url, namely:

    Dockerfile
    job.yaml
    redis-pod.yaml
    redis-service.yaml
    rediswq.py
    worker.py

2) Start minikube:

    >> minikube start

2) Run the redis-pod and redis-svc using:
    This starts the redis server to which we push the queue and from which
    the workers pull the queue items from 

    >> kubectl apply -f redis-pod.yaml
    >> kubectl apply -f redis-service.yaml

3) Create the appropriate docker image for the workers and tag it
    ** This needs to be done in the minikube docker-env for everything to work
    ** Unsurprisingly the minikube environment is different, this can be checked
        using "docker images" before and after the following command
    >> eval $(minikube docker-env)

    >> docker build -t job-wq-2 .
    >> docker tag job-wq-2 abuzarmahmood/job-wq-2

    ** This can probably be done together

4) Change the name for the image in the job.yaml file and 
    change imagePullPolicy to Never so only the local image is pulled 

4) Push items to the work queue as presribed in the tutorial
    ** Note : Multiple items can be insrted using

    >> cat data.txt | redis-cli --pipe

    as per : https://redis.io/topics/mass-insert
    ** Note, this will need a drive/path to be mounted to the pods,
        refer to "Mounting path to job pods" section below

5) Run the job

    >> kubectl apply -f ./job.yaml

    **Note : The worker pods throw the error if the redis queue is not available,
            likely because they can't exit successfully

########################################
-- Mounting path to job pods
** To create read-write allowed path in the job pods
** Refer to <job2.yaml> for an example case which will write a file to the 
    mounted path

1) Start Minikube with a mount. 
    ** The pods generated will then mount the path from within the minikube 
    filesystem (i.e. they don't have access to the host node's filesystem)
    ** The following command will mount "/media/bigdata" from the host node
        to "/test/" in the minikube filesystem

    >> minikube start --v=4 --mount --mount-string="/media/bigdata:/test/"

2) Add "volumes" to spec.template.spec, and add "volumeMounts" to 
    spec.template.spec.containers
    ** This will mount the hostPath from the minikube filesystem to the 
        mountPath in the pod filesystem

REFERENCES:
1) https://groups.google.com/g/kubernetes-users/c/v2806ezEdPk
2) https://dev.to/coherentlogic/learn-how-to-mount-a-local-drive-in-a-pod-in-minikube-2020-3j48
3) https://kubernetes.io/docs/concepts/storage/volumes/#hostpath
