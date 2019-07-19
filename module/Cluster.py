from dask.distributed import Client
from dask_jobqueue import SLURMCluster


class Cluster:
    def __init__(self):
        print("Start Cluster")
        self.cluster = SLURMCluster(
            memory='16g',
            processes=1,
            cores=1,
            death_timeout=200,
            walltime="168:00:00",
            job_extra=['--partition=Sibirien'])
        self.cluster.start_workers(25)
        self.cli = Client(self.cluster.scheduler.address)

    def close(self):
        self.cluster.close()
