from tsfresh.utilities.distribution import DistributorBaseClass
from functools import partial

class SlurmDaskDistributor(DistributorBaseClass):
    """
    Distributor using a dask cluster, meaning that the calculation is spread over a cluster
    """

    def __init__(self, cluster):
        """
        Sets up a distributor that connects to a Dask Scheduler to distribute the calculaton of the features
        :param address: the ip address and port number of the Dask Scheduler
        :type address: str
        """
        self.cluster = cluster
        from distributed import Client

        self.client = Client(cluster)

    def calculate_best_chunk_size(self, data_length):
        """
        Uses the number of dask workers in the cluster (during execution time, meaning when you start the extraction)
        to find the optimal chunk_size.
        :param data_length: A length which defines how many calculations there need to be.
        :type data_length: int
        """
        n_workers = len(self.client.scheduler_info()["workers"])
        chunk_size, extra = divmod(data_length, n_workers * 5)
        if extra:
            chunk_size += 1
        return chunk_size

    def distribute(self, func, partitioned_chunks, kwargs):
        """
        Calculates the features in a parallel fashion by distributing the map command to the dask workers on a cluster
        :param func: the function to send to each worker.
        :type func: callable
        :param partitioned_chunks: The list of data chunks - each element is again
            a list of chunks - and should be processed by one worker.
        :type partitioned_chunks: iterable
        :param kwargs: parameters for the map function
        :type kwargs: dict of string to parameter
        :return: The result of the calculation as a list - each item should be the result of the application of func
            to a single element.
        """

        result = self.client.gather(self.client.map(partial(func, **kwargs), partitioned_chunks))
        return [item for sublist in result for item in sublist]

    def close(self):
        """
        Closes the connection to the Dask Scheduler
        """
        self.client.close()
