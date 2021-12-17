def run(dataset, settings, model, *args):
    loss, _ = model.evaluate(dataset.x_validation, dataset.y_validation, verbose=0)
    return loss
