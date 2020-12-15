gcloud config set project poc-salon-data

CLUSTER_NAME="cluster-dask-vs-koalas"
REGION="europe-west1"
ZONE="europe-west1-b"
gcloud beta dataproc clusters create ${CLUSTER_NAME} \
--region ${REGION} \
--master-machine-type n1-standard-16 \
--worker-machine-type n1-standard-16 \
--num-workers 2 \
--image-version preview \
--zone ${ZONE} \
--initialization-actions gs://dask-vs-koalas/dask.sh \
--metadata dask-runtime=yarn \
--optional-components JUPYTER \
--enable-component-gateway \
--properties=dataproc:dataproc.logging.stackdriver.job.driver.enable=true,dataproc:dataproc.logging.stackdriver.enable=true,dataproc:jobs.file-backed-output.enable=true,dataproc:dataproc.logging.stackdriver.job.yarn.container.enable=true,dataproc:dataproc.logging.stackdriver.enable=true

