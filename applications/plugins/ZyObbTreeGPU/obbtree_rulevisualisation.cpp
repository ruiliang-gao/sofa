#include "obbtree_rulevisualisation.h"
#include "ui_obbtree_rulevisualisation.h"

using namespace sofa::component::collision;


ObbTree_RuleVisualisation::~ObbTree_RuleVisualisation()
{
    delete ui;
}

void ObbTree_RuleVisualisation::updateRules(const std::vector<FakeGripping_Event_Container> &events, const int &active, const int &previous)
{
    ui->textEdit->clear();
	ui->textEdit->setTextColor(Qt::black);
	ui->textEdit->append(QString("Anzahl der Regeln: %1. Active: %2.").arg(events.size()).arg(active));
    for (size_t i = 0; i < events.size(); i++) {
        if (i == active) {
            ui->textEdit->setTextColor(Qt::blue);
        } /*else if (i == previous)  {
            ui->textEdit->setTextColor(Qt::green);
        }  */else {
            ui->textEdit->setTextColor(Qt::gray);
        }
        ui->textEdit->append(QString("Regel %1: %2 - %3")
			.arg(i)
			.arg(events.at(i).activeModel.c_str())
			.arg(events.at(i).leadingModel.c_str())
			);
    }
}
