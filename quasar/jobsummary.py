class JobSummary(object):

    def __init__(
        self,
        resources=None,
        attributes=None,
        jobs=None,
        ):

        self.resources = resources if resources else {}
        self.attributes = attributes if attributes else {}
        self.jobs = jobs

    @staticmethod
    def sum(jobs):

        resources = {}
        for job in jobs:
            for k, v in job.items():
                resources[k] = v + resources.get(k, 0)

        return JobSummary(
            resources=resources,
            jobs=jobs)

    @property
    def leaf_jobs(self): 
        if self.jobs is None:
            return [self]
        else:
            jobs = []
            for job in self.jobs:
                jobs += job.leaf_jobs
            return jobs
