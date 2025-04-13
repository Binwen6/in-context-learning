def evaluate(model, test_data):
    model.eval()
    losses = []
    with torch.no_grad():
        for X, Y, x_query, y_query in test_data:
            with autocast():
                input_seq = torch.cat([torch.cat([X, Y.unsqueeze(-1)], dim=-1),
                                      torch.cat([x_query.unsqueeze(0), torch.zeros(1, 1)], dim=-1)], dim=0).to("cuda")
                output = model(input_seq.unsqueeze(0))[:, -1, :]
                loss = torch.nn.functional.mse_loss(output, y_query.to("cuda"))
            losses.append(loss.item())
    return np.mean(losses)

print("Linear:", evaluate(model, test_linear))
print("Gaussian:", evaluate(model, test_gaussian))
print("Dynamical:", evaluate(model, test_dynamical))
