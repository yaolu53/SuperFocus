from catalyst import dl, metrics, utils

class CustomRunner(dl.Runner):
    
    def on_loader_start(self, runner):
        super().on_loader_start(runner)
        self.meters = {
            key: metrics.AdditiveMetric(compute_on_call=False)
            for key in ["loss"]
        }

    def handle_batch(self, batch):
        x, y = batch
        x = x.view(len(x), -1)
        y_pred = self.model(x)
        loss = self.criterion(y_pred, y)
        self.batch = {
            "features": x,
            "targets": y,
            "preds": y_pred,
        }
        self.batch_metrics.update({"loss": loss})
        
        for key in ["loss"]:
            self.meters[key].update(self.batch_metrics[key].item(), self.batch_size)

        '''
        if self.is_train_loader:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        '''
            
    def on_loader_end(self, runner):
        for key in ["loss"]:
            self.loader_metrics[key] = self.meters[key].compute()[0]
        super().on_loader_end(runner)
    
    