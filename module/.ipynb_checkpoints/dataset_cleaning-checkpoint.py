def removeParticipants(df):
    return df.select(lambda x: not re.search(arg+'_*', x), axis=0)
def test(bigdf, config):

    pass
