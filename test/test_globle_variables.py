import tf_encrypted as tfe


with tfe.Session() as sess:
    sess.run(tfe.global_variables_initializer(),tag='init')

with tfe.Session() as sess:
    sess.run(tfe.global_variables_initializer(),tag='init')