from dask_jobqueue import SLURMCluster
from datetime import datetime
from time import sleep
from module import printing


def start_cluster(workers=2, processes=4, cores=4, rate=0.9):
    cluster = SLURMCluster(
        memory='16g',
        processes=processes,
        cores=cores,
        death_timeout=200,
        walltime="20:00:00",
        job_extra=['--partition=All'])

    cluster.start_workers(workers)
    print(cluster)
    print(cluster.scheduler)
    print(cluster.dashboard_link)
    print("")

    running_jobs = 0
    total_jobs = 1000

    while running_jobs < int(total_jobs*rate):
        sleep(2)
        total_jobs = len(cluster.pending_jobs) + len(cluster.running_jobs)
        running_jobs = len(cluster.running_jobs)
        print("", datetime.now().strftime("%a, %d %B %Y %I:%M:%S"))
        print("", cluster)

    printing.printing_status(cluster)
    return cluster

def single_arg(arg, size):
    return [arg for _ in range(size)]