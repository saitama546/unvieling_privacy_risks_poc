
```text
unveiling_privacy_risks_poc/
├── configs/
│   └── experiments/
├── scripts/
├── src/
│   ├── attacks/
│   ├── datasets/
│   ├── defenses/
│   ├── federated/
│   ├── metrics/
│   ├── models/
│   ├── peft/
│   └── utils/
└── results/

To install the requirements, use the requirements.txt file.
pip3 install -r requirements.txt

1. **Step 1: Config loader + seed setup**

2. **Step 2: Dataset factory + client data split**

3. **Step 3: Model factory from config**

4. **Step 4: PEFT module setup**

5. **Step 5: FL server and client classes**

6. **Step 6: FedSGD client gradient computation**

7. **Step 7: Server receives and stores client gradients**

8. **Step 8: Attack batch selection and gradient extraction check**

9. **Step 9: Gradient-only reconstruction attack**

10. **Step 10: CLIP/text-guided reconstruction attack**

11. **Step 11: Run four attack modes**

* gradient only
* gradient + correct query
* gradient + wrong query
* text only

12. **Step 12: Metrics module**

* PSNR
* SSIM
* LPIPS
* gradient loss
* CLIP similarity

13. **Step 13: Save reconstructed images and results**

14. **Step 14: Test-set accuracy evaluation**

15. **Step 15: Multi-client FedSGD experiment**

16. **Step 16: FedAvg extension**

17. **Step 17: Defense mechanisms**

18. **Step 18: Multiple seeds and result aggregation**
