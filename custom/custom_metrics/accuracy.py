def run(dataset, settings, model, *args):
    loss, acc = model.evaluate(dataset.x_validation, dataset.y_validation, verbose=0)
    return acc
