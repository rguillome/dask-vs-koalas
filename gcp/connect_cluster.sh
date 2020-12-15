CLUSTER_NAME="cluster-dask-vs-koalas"
ZONE="europe-west1-b"

gcloud compute ssh ${CLUSTER_NAME}-m --zone ${ZONE}