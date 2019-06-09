#ifndef OBBTREE_RULEVISUALISATION_H
#define OBBTREE_RULEVISUALISATION_H

#include <QtGlobal>
#if QT_VERSION >= 0x050000
    #include <QtWidgets/QDialog>
#else
    #include <QtGui/QDialog>
#endif

#include "ObbTreeGPUCollisionDetection_Threaded.h"

#include "ui_obbtree_rulevisualisation.h"
/*namespace Ui {
class ObbTree_RuleVisualisation;
}*/

class ObbTree_RuleVisualisation : public QDialog
{
    Q_OBJECT

public:
    explicit ObbTree_RuleVisualisation(QWidget *parent = 0)
        : QDialog(parent)
        , ui(new Ui::ObbTree_RuleVisualisation)
    {
        ui->setupUi(this);
        this->setWindowTitle("Test Framework");
    }
    ~ObbTree_RuleVisualisation();


	void updateRules(const std::vector<sofa::component::collision::FakeGripping_Event_Container> &events, const int &active, const int &previous);
private:
    Ui::ObbTree_RuleVisualisation *ui;

};

#endif // OBBTREE_RULEVISUALISATION_H
