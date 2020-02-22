import { NgModule } from '@angular/core';
import { Routes, RouterModule } from '@angular/router';

import { ClassifierComponent } from "./components/classifier/classifier.component";

// Define the routes
const routes: Routes = [
    { 
        path: "disaster-response/classifier",
        component: ClassifierComponent,
    },
    {
        path: "**",
        redirectTo: "disaster-response/classifier",
        pathMatch: "full",
    }
];


@NgModule({
    imports: [RouterModule.forRoot(routes)],
    exports: [RouterModule]
})
export class AppRoutingModule { }
