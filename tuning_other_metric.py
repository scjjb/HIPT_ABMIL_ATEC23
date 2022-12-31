from ray import tune

tuner = tune.Tuner.restore(
        path="~/ray_results/test_run"
    )

results = tuner.fit()

best_trial = results.get_best_result("loss", "min","all")
print("best trial:", best_trial)
print("Best trial config: {}".format(best_trial.config))
print("Best trial final loss: {}".format(best_trial.metrics["loss"]))
print("Best trial final auc: {}".format(best_trial.metrics["auc"]))
print("Best trial final acuracy: {}".format(best_trial.metrics["accuracy"]))
