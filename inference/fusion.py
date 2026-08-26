def fusion_predict(image_label, image_conf, raman_label, raman_conf):
    if ((image_label == "plastic" and raman_label in ["PE","PET"]) or
        (image_label == "lipid" and raman_label == "lipid")):
        return "Prediction Reliable"
    return "Prediction Unreliable"