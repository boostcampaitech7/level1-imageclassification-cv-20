: << "END"
variable3="False"
variable2="False"
variable1="3e-4"
python train_clip.py "$variable1" "$variable2" "$variable3"
END

variable3="True" #multyprompt
variable2="False" #textFrozen
variable1="1e-5"
python train_clip.py "$variable1" "$variable2" "$variable3"


